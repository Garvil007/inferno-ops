# InfernoOps

An autonomous AI ops agent for a simulated immersion-cooled AI data center. A Streamlit dashboard streams live per-GPU telemetry, an LLM agent (Claude) detects thermal throttling, root-causes it, and takes corrective action — all visible in real time, with a chat panel for ad-hoc questions and an MCP server exposing the same tools to any MCP client.

## Quickstart

Three commands, after cloning the repo and `cd`-ing into it:

```bash
pip install -e ".[dev]"
cp .env.example .env   # then edit .env and set ANTHROPIC_API_KEY
streamlit run app.py
```

Without an API key, the dashboard still runs — live telemetry, chart, health, PUE, and the manual throttle-injection button all work — but the agent decision log and chat panel show "Agent unavailable" instead of real Claude output.

## Architecture

```
simulator.py   RackSimulator — pure, seeded random-walk GPU telemetry (temp, clock, power, flow)
telemetry.py   Shared TelemetrySnapshot / EventRecord dataclasses
tools.py       Four pure tool functions over a telemetry buffer + simulator:
                 read_rack_metrics, detect_throttle_event, generate_rca, adjust_flow_rate
config.py      InfernoConfig — single source of truth for every threshold/tunable
agent.py       Claude tool-use loop (monitoring cycle + chat Q&A), driving the tools
               above via the Anthropic Messages API; a guardrail refuses
               adjust_flow_rate on a GPU that isn't actually throttled
mcp_server.py  Exposes the same four tools.py functions over MCP (FastMCP),
               with its own live simulator/buffer — no Streamlit dependency
dashboard.py   Pure Streamlit-free helpers: color-coded health status, PUE
               calculation, chart-data shaping
app.py         Streamlit entrypoint — wires everything together: live-refresh
               fragment, rolling buffer, agent decision log, chat, inject button
```

`app.py` and `mcp_server.py` are the two entrypoints; everything else is a plain importable module with no UI or network code, so it's directly unit-testable.

## Running the tests

```bash
pytest
```

## Running the MCP server

```bash
python -m inferno_ops.mcp_server
```

Runs on stdio. To inspect it interactively:

```bash
npx @modelcontextprotocol/inspector python -m inferno_ops.mcp_server
```

## Linting, formatting, type-checking

```bash
ruff check .
black --check .
mypy src/ app.py
```

## Demo tips

- The "🔥 Inject throttle now" button (pick a GPU, click it) forces an immediate throttle and triggers an agent cycle right away, instead of waiting for the next refresh tick — useful for a live demo.
- The chat panel answers questions ("why did GPU 3 throttle?", "what's the current flow rate?") using the exact same tools and live buffer as the monitoring cycle — never a separate data path.
