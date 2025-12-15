# Multi-Agent Customer Service System

This repo runs a small multi-agent demo using FastAPI services plus an MCP server.

## Quick start

```bash
pip install -r requirements.txt
python unified_demo.py
```

The script starts the MCP server and the router/data/support/billing agents, waits for their health checks, and then runs five sample prompts end-to-end while printing the router responses and A2A logs.
