"""Connect to the Tavily, Aviationstack, and weather MCP servers."""

import asyncio
import selectors
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlencode

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.group import ClientGroup
from fastmcp.client.transports import StdioTransport
from langchain.mcp import MCPAdapter

PROJECT_DIR = Path(__file__).resolve().parent
AVIATIONSTACK_DIR = PROJECT_DIR / "aviationstack-mcp"
WEATHER_SERVER_SCRIPT = PROJECT_DIR / "custom-weather-mcp-server.py"

# Load the keys from the project-specific env files, while preserving values
# already supplied by the process environment.
load_dotenv(PROJECT_DIR / ".env", override=False)
load_dotenv(AVIATIONSTACK_DIR / ".env", override=False)

search_tool = None
aviation_tools = {}
weather_tools = {}

import asyncio
import selectors
import threading

# Persistent event loop
_mcp_loop = asyncio.SelectorEventLoop(
    selectors.SelectSelector()
)


def _start_mcp_loop():
    asyncio.set_event_loop(_mcp_loop)
    _mcp_loop.run_forever()


_mcp_thread = threading.Thread(
    target=_start_mcp_loop,
    name="mcp-event-loop",
    daemon=True
)

_mcp_thread.start()


def run_mcp(coro):
    """Execute MCP coroutine on a persistent event loop."""

    future = asyncio.run_coroutine_threadsafe(
        coro,
        _mcp_loop
    )

    return future.result()

def create_mcp_group() -> ClientGroup:
    """Create independent FastMCP clients for each configured server."""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is not set in the environment or .env")

    aviationstack_api_key = (
        os.getenv("AVIATION_STACK_API_KEY")
        or os.getenv("AVIATIONSTACK_API_KEY")
        or os.getenv("aviationstack_api_key")
    )
    if not aviationstack_api_key:
        raise RuntimeError(
            "AVIATIONSTACK_API_KEY is not set in the environment or .env"
        )
    if not AVIATIONSTACK_DIR.is_dir():
        raise FileNotFoundError(
            f"Aviationstack MCP project was not found: {AVIATIONSTACK_DIR}"
        )
    if not WEATHER_SERVER_SCRIPT.is_file():
        raise FileNotFoundError(
            f"Weather MCP server was not found: {WEATHER_SERVER_SCRIPT}"
        )

    tavily_url = "https://mcp.tavily.com/mcp/?" + urlencode(
        {"tavilyApiKey": tavily_api_key}
    )
    aviationstack_transport = StdioTransport(
        command="uv",
        args=["--directory", str(AVIATIONSTACK_DIR), "run", "aviationstack-mcp"],
        env={"AVIATION_STACK_API_KEY": aviationstack_api_key},
        cwd=str(AVIATIONSTACK_DIR),
    )
    weather_transport = StdioTransport(
        command=os.sys.executable,
        args=[str(WEATHER_SERVER_SCRIPT)],
        cwd=str(PROJECT_DIR),
    )

    return ClientGroup(
        {
            "tavily": Client(tavily_url),
            "aviationstack": Client(aviationstack_transport),
            "weather": Client(weather_transport),
        }
    )


async def initialize_mcp() -> None:
    """Discover tools once and expose them in the module-level registries."""
    global search_tool, aviation_tools, weather_tools

    if search_tool is not None and aviation_tools and weather_tools:
        return

    group = create_mcp_group()
    async with MCPAdapter(group) as adapter:
        tools = await adapter.list_tools()

    search_tool = next(
        (tool for tool in tools if tool.name.endswith("_tavily_search")), None
    )
    if search_tool is None:
        available = ", ".join(tool.name for tool in tools) or "none"
        raise RuntimeError(
            f"Tavily search tool was not found. Available tools: {available}"
        )

    aviation_tools = {
        tool.name: tool for tool in tools if tool.name.startswith("aviationstack_")
    }
    weather_tools = {
        tool.name: tool for tool in tools if tool.name.startswith("weather_")
    }
    required_weather_tools = {"weather_get_current_weather", "weather_get_forecast"}
    missing_weather_tools = required_weather_tools - weather_tools.keys()
    if missing_weather_tools:
        available = ", ".join(sorted(weather_tools)) or "none"
        raise RuntimeError(
            "Weather tools were not found: "
            f"{', '.join(sorted(missing_weather_tools))}. Available: {available}"
        )


async def tavily_mcp_search(query: str):
    await initialize_mcp()
    result = await search_tool.ainvoke({"query": query})
    return result


async def aviation_mcp_call(tool_name: str, tool_args: dict = None):
    await initialize_mcp()

    # Support both short and fully qualified names
    if not tool_name.startswith("aviationstack_"):
        tool_name = f"aviationstack_{tool_name}"

    tool = aviation_tools.get(tool_name)

    if not tool:
        raise ValueError(f"Aviation tool not found: {tool_name}")

    return await tool.ainvoke(tool_args or {})


async def get_airports():

    await initialize_mcp()

    tool = aviation_tools.get("list_airports")

    if not tool:
        return "Airport tool unavailable"

    result = await tool.ainvoke({})

    return result


async def get_airlines():

    await initialize_mcp()

    tool = aviation_tools.get("list_airlines")

    if not tool:
        return "Airline tool unavailable"

    result = await tool.ainvoke({})

    return result

async def initialize_weather_tools():
    """Ensure weather server tools have been discovered."""
    await initialize_mcp()


async def weather_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tools["weather_get_current_weather"].ainvoke({"city": city})


async def forecast_mcp_search(city: str):

    await initialize_weather_tools()

    return await weather_tools["weather_get_forecast"].ainvoke({"city": city})




from langchain_groq import ChatGroq

# LLM
llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

###################################
# Destination Extractor
###################################

def extract_destination(query: str):

    prompt = f"""
    Extract only the destination city or country.

    Query:
    {query}

    Return only destination name.
    """

    response = llm.invoke(prompt)

    return response.content.strip()

async def main() -> None:
    await initialize_mcp()
    print(f"Search Tool: {search_tool.name}")
    print("Aviationstack tools:")
    for name, tool in aviation_tools.items():
        print(f"- {name}: {tool.description}")


if __name__ == "__main__":
    asyncio.run(main())
