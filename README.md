# Multi-Agent Customer Service System (A2A + MCP)

This project demonstrates an agent-to-agent (A2A) coordination layer backed by an MCP server and SQLite database. Agents use an OpenAI LLM for intent parsing and response writing.

## Setup
1. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Initialize the database**
   ```bash
   python scripts/init_db.py
   ```

3. **Run the end-to-end demo (starts MCP + A2A servers automatically)**
   ```bash
   python end_to_end_demo.py
   ```
   Logs are written to `demos/output/run.log`.

4. **(Optional) Start the A2A HTTP server manually**
   ```bash
   uvicorn src.a2a_http:app --host 127.0.0.1 --port 8000
   ```

## Environment Variables
- `OPENAI_API_KEY`: OpenAI key (required for real LLM calls; fallback responses used if missing)
- `OPENAI_MODEL`: Model name (default: `gpt-5`)
- `OPENAI_MODE`: `responses` (default) or `chat`
- `DB_PATH`: SQLite file path (default: `./support.db`)
- `A2A_BASE_URL`: Base URL for A2A endpoints (default: `http://127.0.0.1:8000`)

## Components
- **MCP Server** (`mcp_server/server.py`): Exposes tools backed by SQLite.
- **Agents** (`src/agents/`): Router, Data, and Support specialists.
- **A2A HTTP** (`src/a2a_http.py`): FastAPI JSON-RPC 2.0 endpoints and agent cards.
- **Demo** (`end_to_end_demo.py`): Runs the five required customer scenarios and logs all interactions.
