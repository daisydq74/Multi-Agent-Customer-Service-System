"""MCP client wrapper for interacting with FastMCP tools."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional, Union

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

logger = logging.getLogger(__name__)


class MCPClient:
    """Async MCP client that manages a stdio subprocess."""

    def __init__(self, command: Optional[Union[str, List[str]]] = None) -> None:
        self.command = command or ["python", "-m", "mcp_server.server"]
        self._client = None
        self._session: Optional[ClientSession] = None
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        """Start the MCP stdio client if not already started."""
        async with self._lock:
            if self._session is not None:
                return
            cmd = self.command or "python"
            if isinstance(cmd, (list, tuple)):
                command = cmd[0]
                args = list(cmd[1:])
            else:
                command = cmd
                args = ["-m", "mcp_server.server"]

            params = StdioServerParameters(
                command=command,
                args=args,
                env={"DB_PATH": os.getenv("DB_PATH", "./support.db")},
            )
            self._client = await stdio_client(params)
            self._session = ClientSession(self._client)
            await self._session.__aenter__()
            logger.info("MCP client started with command=%s", self.command)

    async def close(self) -> None:
        """Close the MCP client session."""
        if self._session is not None:
            await self._session.__aexit__(None, None, None)
            self._session = None
        if self._client is not None:
            await self._client.aclose()
            self._client = None
            logger.info("MCP client closed")

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Invoke a tool by name and parse the return content."""
        if self._session is None:
            await self.start()
        if self._session is None:
            raise RuntimeError("MCP session failed to start")
        logger.info("Calling MCP tool=%s args=%s", tool_name, arguments)
        results = await self._session.call_tool(tool_name, arguments)
        parsed: List[Any] = []
        for result in results:
            for content in getattr(result, "content", []) or []:
                if hasattr(content, "data") and content.data is not None:
                    parsed.append(content.data)
                elif hasattr(content, "json") and content.json is not None:
                    parsed.append(content.json)
                elif hasattr(content, "text") and content.text is not None:
                    text_val = content.text
                    try:
                        parsed.append(json.loads(text_val))
                    except Exception:
                        parsed.append(text_val)
        if not parsed:
            return None
        if len(parsed) == 1:
            return parsed[0]
        return parsed


shared_mcp_client = MCPClient()
