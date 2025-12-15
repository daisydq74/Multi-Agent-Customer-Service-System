The multi-agent demo now has clearer guidance for spinning up the MCP server alongside the routing, data, support, and billing services. A lightweight SQLite database seeds realistic customer and ticket records so the agents can generate contextual replies without extra setup.

Running `python demo.py` brings the entire stack online, drives sample customer prompts through the router, and then shuts everything down automatically. This flow makes it easy to observe how agent-to-agent JSON-RPC calls and MCP tool invocations fit together in a single run.
