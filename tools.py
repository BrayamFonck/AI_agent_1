from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.tools import tool
from datetime import datetime
import traceback


# ── HERRAMIENTA DE GUARDADO ────────────────────────────────────────────────────

# El decorador @tool convierte la función directamente en una herramienta de LangChain
# El agente usará el nombre de la función y su docstring para saber cuándo usarla
@tool
def save_tool(data: str, filename: str = "research_output.txt") -> str:
    """Saves structured research data to a text file.
    Args:
        data: The content to save
        filename: The name of the file to save to
    """
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)

        return f"Data successfully saved to {filename}"
    except PermissionError:
        return (
            f"[save_tool_error] No hay permisos para escribir en '{filename}'. "
            "Verifica permisos de carpeta/archivo."
        )
    except OSError as exc:
        return f"[save_tool_error] Error de sistema al guardar archivo: {exc}"
    except Exception as exc:
        return (
            "[save_tool_error] Error inesperado al guardar. "
            f"Detalle: {exc}\n{traceback.format_exc()}"
        )


# ── HERRAMIENTA DE BÚSQUEDA WEB ───────────────────────────────────────────────

@tool
def search_tool(query: str) -> str:
    """Search the web for information using DuckDuckGo.
    Args:
        query: The search terms to look for
    """
    if not query or not query.strip():
        return "[search_tool_error] La consulta está vacía. Proporciona un texto para buscar."

    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except ImportError as exc:
        return (
            "[search_tool_error] Falta la dependencia 'ddgs'. "
            "Instala con: pip install -U ddgs. "
            f"Detalle: {exc}"
        )
    except Exception as exc:
        return (
            "[search_tool_error] Falló la búsqueda web (posible red/bloqueo/rate-limit). "
            f"Detalle: {exc}\n{traceback.format_exc()}"
        )


# ── HERRAMIENTA DE WIKIPEDIA ──────────────────────────────────────────────────

# Wikipedia no necesita @tool porque WikipediaQueryRun ya es una Tool de LangChain
try:
    import wikipedia

    api_wrapper = WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=1, doc_content_chars_max=100)
    wiki_tool = WikipediaQueryRun(api_wrapper=api_wrapper)
except Exception:
    wiki_tool = None
    