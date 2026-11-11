"""Exercise the local weather MCP server over stdio.

Run from the project environment, for example:
    uv run python testing-weather-mcp-server.py London
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from fastmcp import Client
from fastmcp.client.transports import StdioTransport


PROJECT_DIR = Path(__file__).resolve().parent
SERVER_SCRIPT = PROJECT_DIR / "custom-weather-mcp-server.py"

async def test_weather_server(city: str) -> None:
    
    print(f"Testing weather MCP server: {SERVER_SCRIPT}")

    # The server loads its .env from the working directory.
    load_dotenv(PROJECT_DIR / ".env")
    if not os.getenv("OPENWEATHER_API_KEY"):
        raise RuntimeError("OPENWEATHER_API_KEY is missing from .env or environment")
    if not SERVER_SCRIPT.is_file():
        raise FileNotFoundError(f"Weather MCP server not found: {SERVER_SCRIPT}")

    transport = StdioTransport(
        command=sys.executable,
        args=[str(SERVER_SCRIPT)],
        cwd=str(PROJECT_DIR),
    )

    async with Client(transport) as client:
        tools = await client.list_tools()
        print("Available tools:")
        for tool in tools:
            print(f"- {tool.name}: {tool.description}")

        tool_names = {tool.name for tool in tools}
        required_tools = {"get_current_weather", "get_forecast"}
        missing_tools = required_tools - tool_names
        if missing_tools:
            raise RuntimeError(
                "Weather server is missing expected tools: "
                + ", ".join(sorted(missing_tools))
            )

        for tool_name in ("get_current_weather", "get_forecast"):
            print(f"\n{tool_name}({city!r}):")
            result = await client.call_tool(tool_name, {"city": city})
            print(result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test the weather MCP server over stdio.")
    parser.add_argument("city", nargs="?", default="London", help="City to query")
    args = parser.parse_args()
    asyncio.run(test_weather_server(args.city))


if __name__ == "__main__":
    main()
