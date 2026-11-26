from __future__ import annotations

import json

import pytest
from mcp.shared.memory import create_connected_server_and_client_session

from alisa.mcp_server import mcp


@pytest.fixture
async def client_session():
    async with create_connected_server_and_client_session(
        mcp._mcp_server, raise_exceptions=True
    ) as session:
        yield session


@pytest.mark.anyio
async def test_list_tools_returns_both_alisa_tools(client_session):
    result = await client_session.list_tools()
    names = {tool.name for tool in result.tools}
    assert names == {"search_flights", "search_airline_rules"}


@pytest.mark.anyio
async def test_search_flights_tool_returns_ranked_top3(client_session):
    result = await client_session.call_tool(
        "search_flights", {"from_city": "MOW", "to_city": "AER"}
    )
    assert result.isError is not True
    payload = json.loads(result.content[0].text)
    assert "top3" in payload
    assert payload["candidates_count"] >= len(payload["top3"])


@pytest.mark.anyio
async def test_search_flights_tool_missing_required_arg_is_protocol_error(
    client_session,
):
    # MCP сообщает об ошибке валидации через isError.
    result = await client_session.call_tool("search_flights", {"to_city": "AER"})
    assert result.isError is True
    assert "from_city" in result.content[0].text


@pytest.mark.anyio
async def test_search_airline_rules_tool_finds_baggage_section(client_session):
    result = await client_session.call_tool(
        "search_airline_rules", {"query": "сколько багажа можно провезти"}
    )
    assert result.isError is not True
    payload = json.loads(result.content[0].text)
    assert len(payload["rules"]) >= 1
    assert any("багаж" in rule["text"].lower() for rule in payload["rules"])


@pytest.mark.anyio
async def test_search_airline_rules_tool_narrows_by_airline(client_session):
    result = await client_session.call_tool(
        "search_airline_rules", {"query": "провоз лыж", "airline": "Победа"}
    )
    payload = json.loads(result.content[0].text)
    assert all(rule["airline"] == "Победа" for rule in payload["rules"])


@pytest.fixture
def anyio_backend():
    return "asyncio"
