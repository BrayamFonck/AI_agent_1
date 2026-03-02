# importaciones
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_groq import ChatGroq
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import HumanMessage
from tools import search_tool, save_tool
from langchain.agents import create_agent
import os
import traceback

# cargar variables de entorno
load_dotenv()

# Estructura de la respuesta esperada
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]


def is_tool_use_failed_error(exc: Exception) -> bool:
    message = str(exc)
    return "tool_use_failed" in message or "Failed to call a function" in message

def build_llm() -> ChatGroq:
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError(
            "Falta GROQ_API_KEY en variables de entorno. "
            "Configúrala en tu archivo .env."
        )

    model_name = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    return ChatGroq(
        model=model_name,
        temperature=0,
    )


def extract_text_from_agent_result(result: dict) -> str:
    messages = result.get("messages")
    if not messages:
        raise ValueError("El agente no devolvió mensajes en la respuesta.")

    last_msg = messages[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if content is None:
        raise ValueError("La respuesta del agente no contiene texto utilizable.")

    return str(content)


def extract_text_from_llm_response(response) -> str:
    content = response.content if hasattr(response, "content") else str(response)

    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )

    if content is None:
        raise ValueError("El LLM devolvió una respuesta sin contenido utilizable.")

    return str(content)


def fallback_without_tools(llm: ChatGroq, parser: PydanticOutputParser, query: str) -> str:
    search_result = search_tool.invoke({"query": query})
    format_instructions = parser.get_format_instructions()

    fallback_prompt = (
        "Eres un asistente de investigación. "
        "No uses function-calling. "
        "Con la siguiente consulta del usuario y evidencia, responde SOLO con JSON válido.\n\n"
        f"Consulta: {query}\n\n"
        f"Evidencia de búsqueda: {search_result}\n\n"
        f"Formato requerido:\n{format_instructions}\n"
    )

    response = llm.invoke([HumanMessage(content=fallback_prompt)])
    return extract_text_from_llm_response(response)


def main() -> None:
    parser = PydanticOutputParser(pydantic_object=ResearchResponse)

    system_prompt = """
You are a research assistant.
Use tools when needed.
When you finish, return a concise final answer.
"""

    tools = [search_tool, save_tool]

    try:
        llm = build_llm()
        agent = create_agent(
            model=llm,
            tools=tools,
            system_prompt=system_prompt,
        )
    except Exception as exc:
        print(f"Error al inicializar el agente/LLM: {exc}")
        print(traceback.format_exc())
        return

    try:
        query = input("Qué puedo buscar por ti hoy? ").strip()
        if not query:
            print("La consulta está vacía. Escribe algo para buscar.")
            return

        output_text = ""
        try:
            result = agent.invoke({
                "messages": [{"role": "user", "content": query}]
            })
            output_text = extract_text_from_agent_result(result)
        except Exception as exc:
            if is_tool_use_failed_error(exc):
                print(
                    "Advertencia: el modelo actual falló en tool-calling "
                    "(tool_use_failed). Aplicando fallback sin function-calling..."
                )
                output_text = fallback_without_tools(llm, parser, query)
            else:
                raise

        try:
            structured_response = parser.parse(output_text)
            print("\nRespuesta estructurada:")
            print(structured_response)
        except Exception as exc:
            print(f"Error al parsear la respuesta: {exc}")
            print("Salida cruda del agente:")
            print(output_text)
    except KeyboardInterrupt:
        print("\nEjecución cancelada por el usuario.")
    except Exception as exc:
        print(f"Error durante la ejecución del agente: {exc}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()








