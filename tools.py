from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime
import traceback

# ── ESQUEMAS DE ENTRADA (Ayudan al LLM a no fallar el Tool Calling) ────
class SaveToolInput(BaseModel):
    data: str = Field(description="The exact text content to be saved to the file.")
    filename: str = Field(default="research_output.txt", description="The name of the file to save the data in.")

class SearchToolInput(BaseModel):
    query: str = Field(description="The search query to look up on the web. Keep it simple and without special characters if possible.")

class WikipediaToolInput(BaseModel):
    query: str = Field(description="The topic to search for on Wikipedia.")

# ── HERRAMIENTA DE GUARDADO ────────────────────────────────────────────

@tool("save_tool", args_schema=SaveToolInput)
def save_tool(data: str, filename: str = "research_output.txt") -> str:
    """Saves structured research data or notes to a text file on the local system."""
    try:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

        with open(filename, "a", encoding="utf-8") as f:
            f.write(formatted_text)

        return f"Data successfully saved to {filename}"
    except Exception as exc:
        return f"[save_tool_error] Detailed error: {exc}"


# ── HERRAMIENTA DE BÚSQUEDA WEB ────────────────────────────────────────

@tool("search_tool", args_schema=SearchToolInput)
def search_tool(query: str) -> str:
    """Search the web for current information, news, or general queries."""
    if not query or not query.strip():
        return "[search_tool_error] Empty query."

    try:
        search = DuckDuckGoSearchRun()
        return search.run(query)
    except Exception as exc:
        return f"[search_tool_error] Web search failed. Details: {exc}"


# ── HERRAMIENTA DE WIKIPEDIA ───────────────────────────────────────────

@tool("wikipedia_search_tool", args_schema=WikipediaToolInput)
def wikipedia_search_tool(query: str) -> str:
    """Search Wikipedia for information about historical facts, concepts, or entities."""
    if not query or not query.strip():
        return "[wikipedia_error] Empty query."
        
    try:
        import wikipedia
        api_wrapper = WikipediaAPIWrapper(wiki_client=wikipedia, top_k_results=1, doc_content_chars_max=1000)
        return api_wrapper.run(query)
    except Exception as exc:
        return f"[wikipedia_error] Wikipedia search failed: {exc}"