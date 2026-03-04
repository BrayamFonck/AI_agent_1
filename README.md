# AI Agent 1 - Investigador con LangChain + Groq

Proyecto de consola en Python que ejecuta un agente de investigación con herramientas de búsqueda web y Wikipedia, y devuelve una respuesta estructurada.

## ¿Qué hace este proyecto?

- Recibe una consulta desde consola.
- Investiga usando herramientas (`search_tool` y `wikipedia_search_tool`).
- Guarda resultados usando `save_tool` en archivos `.txt`.
- Intenta devolver una salida estructurada con el modelo `ResearchResponse`:
  - `topic`
  - `summary`
  - `sources`
  - `tools_used`

Además, incluye un flujo de fallback para cuando el modelo no soporta tool-calling.

## Tecnologías principales

- Python
- LangChain
- LangChain Groq (`ChatGroq`)
- Pydantic
- DuckDuckGo Search
- Wikipedia API wrapper
- python-dotenv

## Estructura del proyecto

```text
AI_agent_1/
├── main.py
├── tools.py
├── requirements.txt
├── mundial_2026.txt
├── clasificacion_colombia_mundial_2026.txt
├── oponentes_colombia_mundial_2026.txt
└── uis_estudiantes.txt
```

- `main.py`: flujo principal del agente, parseo estructurado y fallback.
- `tools.py`: definición de herramientas de búsqueda, Wikipedia y guardado.
- `requirements.txt`: dependencias del proyecto.
- `*.txt`: ejemplos de salidas guardadas por `save_tool`.

## Requisitos

- Python 3.10 o superior
- Una API key de Groq

## Instalación

1. Crear entorno virtual (opcional pero recomendado):

```powershell
python -m venv venv
```

2. Activar entorno virtual en Windows PowerShell:

```powershell
venv\Scripts\Activate.ps1
```

3. Instalar dependencias:

```powershell
pip install -r requirements.txt
```

## Configuración

Crea un archivo `.env` en la raíz del proyecto con:

```env
GROQ_API_KEY=tu_api_key_aqui
GROQ_MODEL=llama-3.3-70b-versatile
```

`GROQ_MODEL` es opcional (el código usa ese valor por defecto si no está definido).

## Uso

Ejecuta:

```powershell
python main.py
```

Luego escribe tu consulta cuando aparezca:

```text
¿Qué puedo buscar por ti hoy?
```

## Comportamiento esperado

- Si todo funciona bien, verás una respuesta estructurada en consola.
- Si falla el tool-calling del modelo, se activa un modo fallback que hace búsqueda manual y sigue devolviendo una respuesta estructurada.
- Si el parseo estructurado falla, el sistema muestra la respuesta sin estructurar para no perder información.

## Notas

- `save_tool` guarda texto en modo append, así que los archivos se van acumulando con timestamp.
- Las consultas pueden formularse en español, pero internamente el prompt recomienda evitar caracteres especiales en queries de herramientas para reducir errores de codificación.

## Próximas mejoras sugeridas

- Agregar tests automáticos para funciones de parseo y fallback.
- Normalizar y limpiar `requirements.txt` (hay paquetes potencialmente no usados).
- Añadir logging estructurado en lugar de `print` para entornos de producción.