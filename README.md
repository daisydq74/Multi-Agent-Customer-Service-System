# Multi-Agent Customer Service System

This project runs a small multi-agent customer service demo built with FastAPI microservices plus an MCP server backed by SQLite. A router agent orchestrates requests across specialist data, support, and billing agents, while the MCP server exposes database tools the data agent can call.

## Components
- **MCP server (port 8000):** Provides database-backed tools for customers and tickets.
- **Data agent (port 8011):** Calls MCP tools to fetch/update customer records or assemble reports.
- **Support agent (port 8012):** Crafts empathetic replies and decides when to escalate to billing.
- **Billing agent (port 8013):** Handles escalated billing issues with concise next steps.
- **Router agent (port 8010):** Parses the incoming request, calls the data agent, and orchestrates support/billing responses.

## Quick start
```bash
pip install -r requirements.txt
python demo.py
```
`demo.py` starts all five FastAPI services, waits for their health checks, runs five sample prompts end-to-end, prints the router replies, and then shuts the services down.

## Data storage
The MCP server initializes an on-disk SQLite database (`support.db`) with sample customers and tickets on first run. No manual setup is required, but you can inspect or adjust the seed data via `database_setup.py`.

## Endpoints
Each service exposes a `/health` endpoint for readiness checks and a `/rpc` JSON-RPC endpoint for agent-to-agent messages. The MCP server also serves `/tools/list`, `/tools/call`, and `/events/stream` for tool metadata, execution, and event streaming respectively.
