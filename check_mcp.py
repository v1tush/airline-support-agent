import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def check():
    root = Path(__file__).absolute().parent
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "alisa.mcp_server"],
        cwd=str(root),
        env={**os.environ, "PYTHONPATH": str(root), "ALISA_SEMANTIC": "0"},
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            listed = await session.list_tools()
            good = await session.call_tool(
                "search_flights",
                {"from_city": "MOW", "to_city": "AER", "dates": ["2026-07-23"]},
            )
            bad = await session.call_tool("search_flights", {"to_city": "AER"})
            assert "search_flights" in [t.name for t in listed.tools]
            assert not good.isError and bad.isError
            return {
                "protocol": init.protocolVersion,
                "tools": [t.name for t in listed.tools],
                "good": good.model_dump(),
                "bad": bad.model_dump(),
            }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(check()), ensure_ascii=False, indent=2))
