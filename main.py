# ============================================================
# IMPORTACIONES
# ============================================================

import os           # Lectura de variables de entorno del sistema
import re           # Expresiones regulares para extraer JSON del texto del agente
import json         # Parseo manual de JSON sin depender de una segunda llamada al LLM
import traceback    # Trazas detalladas de errores para facilitar debugging
from dotenv import load_dotenv                          # Carga el archivo .env al entorno
from pydantic import BaseModel, ValidationError         # Modelo de datos con validación automática de tipos
from langchain_groq import ChatGroq                     # Cliente LLM para la API de Groq
from langchain_core.messages import HumanMessage        # Representa un mensaje del usuario
from langchain_core.output_parsers import PydanticOutputParser  # Parser de respaldo basado en Pydantic
from langchain.agents import create_agent               # Crea el agente ReAct con herramientas
from tools import search_tool, save_tool, wikipedia_search_tool
# Herramientas disponibles para el agente:
#   search_tool            → búsqueda web general
#   save_tool              → guarda los hallazgos en disco
#   wikipedia_search_tool  → consulta directa a Wikipedia


# ============================================================
# CARGA DE VARIABLES DE ENTORNO
# ============================================================

load_dotenv()
# NECESARIO: Sin esto, os.getenv() devuelve None aunque las claves
# estén bien definidas en el archivo .env.


# ============================================================
# MODELO DE RESPUESTA ESTRUCTURADA
# ============================================================

class ResearchResponse(BaseModel):
    topic: str             # Tema principal investigado
    summary: str           # Resumen narrativo de los hallazgos
    sources: list[str]     # URLs o referencias de las fuentes consultadas
    tools_used: list[str]  # Herramientas que el agente utilizó durante la investigación

# NECESARIO: Define el "contrato" de salida del sistema.
# Pydantic valida tipos en tiempo de ejecución: si el LLM devuelve
# un número donde se espera una lista, el error es claro e inmediato.
# Este modelo se usa tanto para el parseo manual como para el fallback.


# ============================================================
# CONSTRUCCIÓN DEL CLIENTE LLM
# ============================================================

def build_llm() -> ChatGroq:
    """Construye y valida el cliente LLM antes de usarlo."""

    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "Falta GROQ_API_KEY en variables de entorno. "
            "Configúrala en tu archivo .env."
        )
    # NECESARIO: Falla rápido con mensaje claro.
    # Sin este chequeo, el error aparecería como un 401 HTTP críptico
    # desde la API de Groq, difícil de diagnosticar.

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    # Permite cambiar el modelo editando solo el .env sin tocar el código.
    # El valor por defecto garantiza funcionamiento aunque no esté definida la variable.

    return ChatGroq(
        model=model_name,
        temperature=0,
        # temperature=0 → respuestas deterministas y reproducibles.
        # Esencial para investigación: evita variaciones creativas innecesarias
        # y hace que el JSON generado sea más consistente entre ejecuciones.
    )


# ============================================================
# PARSEO MANUAL DE JSON (evita segunda llamada al LLM)
# ============================================================

def parse_json_response(text: str, parser: PydanticOutputParser) -> ResearchResponse:
    """
    Intenta extraer y validar el JSON de la respuesta del agente.
    Estrategia en 3 niveles, de más estricto a más permisivo:
      1. Busca bloque JSON explícito con regex
      2. Intenta parsear el texto completo como JSON
      3. Delega al PydanticOutputParser de LangChain como último recurso
    
    Esto evita completamente la necesidad de una segunda llamada al LLM
    para estructurar la respuesta.
    """

    # --- Nivel 1: Extracción con regex ---
    # Busca el bloque JSON más largo en el texto, tolerando saltos de línea.
    # re.DOTALL permite que '.' también coincida con '\n'.
    match = re.search(r'\{[\s\S]*\}', text)
    if match:
        try:
            data = json.loads(match.group())
            return ResearchResponse(**data)
            # ResearchResponse(**data) valida tipos vía Pydantic.
            # Si falta un campo o el tipo es incorrecto, lanza ValidationError.
        except (json.JSONDecodeError, ValidationError):
            pass  # Si falla, intenta el siguiente nivel

    # --- Nivel 2: Parseo del texto completo ---
    # Para casos donde el agente devuelve JSON puro sin texto adicional.
    try:
        data = json.loads(text.strip())
        return ResearchResponse(**data)
    except (json.JSONDecodeError, ValidationError):
        pass

    # --- Nivel 3: PydanticOutputParser como último recurso ---
    # LangChain intenta extraer y coercionar el JSON con heurísticas adicionales.
    # Es el más permisivo pero también el menos predecible.
    return parser.parse(text)


# ============================================================
# EXTRACCIÓN DE TEXTO DESDE RESPUESTA DEL AGENTE
# ============================================================

def extract_text_from_agent_result(result: dict) -> str:
    """
    Extrae el texto final de la respuesta del agente LangGraph.
    El agente genera múltiples mensajes intermedios (razonamiento,
    llamadas a herramientas, resultados). Solo nos interesa el último.
    """

    messages = result.get("messages")
    if not messages:
        return ""
    # Devuelve vacío en lugar de lanzar excepción para que el flujo
    # principal decida cómo manejar la ausencia de respuesta.

    last_msg = messages[-1]
    # El ÚLTIMO mensaje siempre contiene la respuesta final al usuario.

    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
    # hasattr() protege contra objetos que no sean instancias de LangChain.

    if isinstance(content, list):
        # Algunos modelos devuelven contenido como lista de bloques mixtos:
        # [{"type": "text", "text": "..."}, {"type": "tool_use", ...}]
        # Este bloque extrae y une solo las partes de texto plano.
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    return str(content) if content else ""


# ============================================================
# FLUJO DE EMERGENCIA: FALLBACK SIN TOOL-CALLING
# ============================================================

def fallback_without_tools(
    llm: ChatGroq,
    parser: PydanticOutputParser,
    query: str
) -> ResearchResponse:
    """
    Plan B cuando el modelo no soporta function-calling (tool_use_failed).
    
    En lugar de crashear, ejecuta la búsqueda manualmente e inyecta
    los resultados como contexto en el prompt. El LLM solo necesita
    razonar y formatear, sin llamar herramientas.
    
    Ventaja: funciona con CUALQUIER modelo, incluso los más básicos.
    """

    print("⚠️  Ejecutando búsqueda manual para el fallback...")
    search_result = search_tool.invoke({"query": query})
    # Ejecutamos la búsqueda externamente porque el modelo no puede
    # invocar herramientas por sí mismo en este flujo.

    format_instructions = parser.get_format_instructions()
    # Obtiene las instrucciones de formato JSON de Pydantic.
    # Le dice exactamente al LLM qué campos debe incluir y de qué tipo.

    fallback_prompt = (
        "Eres un asistente de investigación. "
        "NO uses function-calling ni herramientas. "
        "Basándote SOLO en la evidencia proporcionada, responde "
        "ÚNICAMENTE con JSON válido, sin texto adicional, sin markdown.\n\n"
        f"Consulta del usuario: {query}\n\n"
        f"Evidencia de búsqueda:\n{search_result}\n\n"
        f"Formato JSON requerido:\n{format_instructions}\n"
    )
    # El prompt inyecta directamente los resultados de búsqueda,
    # compensando la falta de herramientas con contexto ya recopilado.
    # "sin markdown" evita que el modelo envuelva el JSON en ```json```.

    response = llm.invoke([HumanMessage(content=fallback_prompt)])

    content = response.content if hasattr(response, "content") else str(response)
    if isinstance(content, list):
        content = "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    # Intenta parsear el JSON de la respuesta del fallback
    return parse_json_response(str(content), parser)


# ============================================================
# DETECCIÓN DE ERRORES DE TOOL-CALLING
# ============================================================

def is_tool_use_failed(exc: Exception) -> bool:
    """
    Identifica si el error es por incompatibilidad del modelo con
    function-calling, para activar el fallback en lugar de crashear.
    
    Algunos modelos de Groq (especialmente los más pequeños) no
    soportan tool-calling y lanzan estos mensajes específicos.
    """
    msg = str(exc).lower()
    return any(keyword in msg for keyword in [
        "tool_use_failed",
        "failed to call a function",
        "function call failed",
        "tool call failed",
    ])
    # Lista extensible: si aparece un nuevo mensaje de error de tool-calling,
    # solo hay que agregar el keyword aquí sin cambiar la lógica principal.


# ============================================================
# FUNCIÓN PRINCIPAL
# ============================================================

def main() -> None:

    # --- Inicialización del LLM ---
    try:
        llm = build_llm()
    except ValueError as exc:
        # ValueError es el error específico que lanza build_llm() si falta la API Key.
        # Lo separamos para dar un mensaje más claro que un error genérico.
        print(f"❌ Error de configuración: {exc}")
        return
    except Exception as exc:
        print(f"❌ Error inesperado al inicializar el LLM: {exc}")
        return

    # --- Parser de respaldo ---
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)
    # Se inicializa aquí (fuera del bloque de ejecución) para reutilizarlo
    # tanto en el flujo principal como en el fallback.


    # --------------------------------------------------------
    # SYSTEM PROMPT: El corazón de la arquitectura de una sola llamada
    # --------------------------------------------------------
    # Este prompt está diseñado para que el agente:
    #   1. Use herramientas activamente (evita respuestas de memoria)
    #   2. Genere consultas compatibles con APIs (sin tildes/caracteres especiales)
    #   3. Persista los resultados con save_tool siempre
    #   4. Devuelva JSON estructurado DIRECTAMENTE → elimina la necesidad de Fase 2
    #
    # La sección JSON al final es crítica: le dice al modelo exactamente
    # qué formato producir, eliminando ambigüedad y reduciendo errores de parseo.

    format_instructions = parser.get_format_instructions()
    # Generamos las instrucciones de formato desde Pydantic para incluirlas
    # en el system_prompt, asegurando consistencia entre el prompt y el parser.

    system_prompt = f"""Eres un experto asistente de investigación con acceso a herramientas de búsqueda.

REGLAS DE USO DE HERRAMIENTAS:
- SIEMPRE usa search_tool o wikipedia_search_tool para buscar información actualizada.
- Nunca respondas de memoria sin consultar primero las herramientas disponibles.
- Formula las consultas a herramientas en INGLÉS o español sin tildes ni caracteres especiales para evitar errores de codificación.
- SIEMPRE usa save_tool al final para guardar un resumen de los hallazgos antes de responder.

PROCESO OBLIGATORIO:
1. Recibe la consulta del usuario.
2. Usa search_tool y/o wikipedia_search_tool para investigar.
3. Usa save_tool para guardar los resultados.
4. Responde con el JSON estructurado indicado abajo.

FORMATO DE RESPUESTA FINAL:
Cuando termines de investigar, responde ÚNICAMENTE con JSON válido.
Sin texto adicional. Sin explicaciones. Sin bloques markdown.

{format_instructions}
"""
    # La inclusión de format_instructions dentro del system_prompt garantiza
    # que el modelo siempre tenga el esquema exacto frente a sí mientras trabaja.


    # --------------------------------------------------------
    # CONSTRUCCIÓN DEL AGENTE
    # --------------------------------------------------------

    tools = [search_tool, save_tool, wikipedia_search_tool]
    # Toolkit completo: búsqueda web + Wikipedia + persistencia de resultados.
    # El orden no importa funcionalmente, pero es buena práctica listar
    # primero las herramientas de búsqueda y al final la de guardado.

    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=system_prompt,
    )
    # Ensambla el agente ReAct (Reason + Act).
    # El ciclo interno es: razonar → elegir herramienta → ejecutar → observar → repetir.
    # El system_prompt define las reglas de ese ciclo.


    # --------------------------------------------------------
    # EJECUCIÓN PRINCIPAL: UNA SOLA LLAMADA AL LLM
    # --------------------------------------------------------

    try:
        query = input("\n¿Qué puedo buscar por ti hoy? ").strip()
        if not query:
            print("La consulta está vacía. Escribe algo para buscar.")
            return
        # .strip() elimina espacios y saltos de línea accidentales.
        # La validación evita una llamada costosa a la API con input vacío.

        print("\n⏳ Investigando... (esto puede tomar unos segundos)")

        output_text = ""

        try:
            # UNA SOLA LLAMADA: el agente investiga Y devuelve JSON en un solo ciclo.
            # Esto es posible gracias al system_prompt que instruye el formato de salida.
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            output_text = extract_text_from_agent_result(result)

            if not output_text:
                raise ValueError("El agente devolvió una respuesta vacía.")

        except Exception as exc:
            if is_tool_use_failed(exc):
                # El modelo no soporta function-calling → activamos el Plan B.
                # En lugar de crashear, ejecutamos el fallback y continuamos.
                print(
                    "\n⚠️  El modelo no soporta tool-calling. "
                    "Activando flujo alternativo..."
                )
                structured_response = fallback_without_tools(llm, parser, query)
                # El fallback ya devuelve un ResearchResponse validado,
                # así que podemos saltar directamente a la impresión.
                print("\n✅ Respuesta (modo fallback):")
                print(structured_response)
                return
            else:
                raise
                # Si el error no es de tool-calling, lo propagamos al
                # except exterior para que se registre con traceback completo.


        # --------------------------------------------------------
        # PARSEO MANUAL: sin segunda llamada al LLM
        # --------------------------------------------------------

        print("✔ Investigación completada. Procesando respuesta...\n")

        try:
            structured_response = parse_json_response(output_text, parser)
            # parse_json_response intenta 3 estrategias en orden de preferencia.
            # Si ninguna funciona, lanza una excepción que capturamos abajo.

            print("✅ Respuesta estructurada:")
            print(f"  📌 Tema:        {structured_response.topic}")
            print(f"  📝 Resumen:     {structured_response.summary}")
            print(f"  🔗 Fuentes:     {', '.join(structured_response.sources)}")
            print(f"  🛠️  Herramientas: {', '.join(structured_response.tools_used)}")

        except Exception as parse_exc:
            # Si el parseo falla completamente, mostramos la respuesta cruda.
            # Esto nunca bloquea al usuario: siempre recibe alguna respuesta.
            print(f"⚠️  No se pudo estructurar la respuesta: {parse_exc}")
            print("\nRespuesta sin estructurar:")
            print(output_text)

    except KeyboardInterrupt:
        # Ctrl+C → cierre limpio sin stack trace aterrador en la consola.
        print("\n\nEjecución cancelada por el usuario.")

    except Exception as exc:
        # Cualquier error no anticipado se registra con traza completa
        # para que el desarrollador pueda diagnosticar exactamente qué falló.
        print(f"\n❌ Error durante la ejecución: {exc}")
        print(traceback.format_exc())


# ============================================================
# PUNTO DE ENTRADA
# ============================================================

if __name__ == "__main__":
    main()
# NECESARIO: main() solo se ejecuta cuando el archivo se corre directamente.
# Si este módulo fuera importado desde otro script, main() NO se ejecutaría,
# evitando efectos secundarios no deseados.