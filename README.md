[![PyPI version](https://badge.fury.io/py/tygent.svg)](https://badge.fury.io/py/tygent)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
# Tygent (Python)

Tygent reshapes unstructured LLM agent plans into structured execution blueprints so downstream tools know which context to fetch and when to run each step. Typed execution graphs[^dag] are the core structure the runtime emits, giving you explicit dependencies, metadata, and prefetch hints that the scheduler can consume for reliable, optimised execution.


## Highlights
- **Structured planner** – normalise free-form plans or framework payloads into typed steps with dependencies, tags, and prefetch directives (`parse_plan`, `ServicePlanBuilder`), exporting context fabric descriptors compatible with Recontext.
- **Context-aware execution** – structured graph metadata[^dag] flows into the scheduler so you can prioritise critical nodes, enforce token budgets, rate-limit API calls, and capture audit traces.
- **Drop-in acceleration** – wrap callables, agents, or plan dictionaries with `tygent.accelerate` to obtain an executor that understands the structured representation.
- **Adaptive workflows** – mutate the structured plan at runtime with `AdaptiveExecutor` rewrite rules for fallbacks, conditional branches, or resource-aware tuning.
- **Multi-agent runtime** – coordinate independent agents through `MultiAgentManager` and a shared `CommunicationBus` while preserving structured plan metadata.
- **Framework patches** – call `tygent.install()` to enable runtime helpers for integrations in `tygent.integrations.*`.
- **Planner adapters** – convert Claude Code, Claude Code CLI, Gemini CLI, and OpenAI Codex planning payloads into scheduler-ready service plans via `tygent.integrations.{claude_code, claude_code_cli, gemini_cli, openai_codex}`.
- **Service bridge + CLI** – the `tyapi` package ships an aiohttp service and CLI that convert third-party plans into the structured format, surface prefetch hints, and benchmark sequential vs accelerated runs.

## Coding-agent integrations

The `tygent.integrations` package includes adapters that turn coding-assistant planning payloads into scheduler-ready `ServicePlan` objects:

- `GeminiCLIPlanAdapter` (`tygent.integrations.gemini_cli`) for Google Gemini CLI plans.
- `ClaudeCodePlanAdapter` (`tygent.integrations.claude_code`) for Anthropic Claude Code traces.
- `ClaudeCodeCLIPlanAdapter` (`tygent.integrations.claude_code_cli`) for Anthropic Claude Code CLI plans.
- `OpenAICodexPlanAdapter` (`tygent.integrations.openai_codex`) for legacy OpenAI Codex workflows.

```python
import asyncio
from tygent.integrations.gemini_cli import GeminiCLIPlanAdapter
from tygent import Scheduler

adapter = GeminiCLIPlanAdapter(gemini_payload)
service_plan = adapter.to_service_plan()
scheduler = Scheduler(service_plan.plan)
results = asyncio.run(scheduler.execute(service_plan.plan))
```

Each adapter also exposes a `patch()` helper that adds a `.to_tygent_service_plan()` method to the upstream planner when the optional dependency is installed, so coding agents can emit Tygent-ready plans in-place.

## Installation

```bash
pip install tygent
```

Development installs (tests, tyapi service, docs tooling) expect Python 3.8+ and the optional extras listed in `pyproject.toml`.

## Quick tour

The snippets below show how Tygent promotes loosely described plans into explicit structures that drive context prefetching and execution control.

### 1. Accelerate a structured plan

```python
import asyncio
from tygent import accelerate

plan = {
    "name": "market_research",
    "steps": [
        {"name": "collect", "func": lambda inputs: {"sources": inputs["query"]}},
        {
            "name": "summarize",
            "func": lambda inputs: f"Summary: {inputs['collect']['sources']}",
            "dependencies": ["collect"],
            "critical": True,
        },
    ],
}


async def main() -> None:
    execute_plan = accelerate(plan)
    result = await execute_plan({"query": "AI funding"})
    print(result["results"]["summarize"])


asyncio.run(main())
```

Passing a plan dictionary produces a scheduler-backed executor that preserves metadata (critical steps, edge mappings, etc.) and returns the scheduler output structure—one of the structured formats Tygent can derive from loosely specified plans.

### 2. Drop-in acceleration for existing code

```python
from tygent import accelerate

@accelerate
def fetch_profile(user_id: str) -> dict:
    # Your existing implementation
    return {"user": user_id}

profile = fetch_profile("abc123")
```

`accelerate` unwraps sync or async callables and inspects attached plans when available (e.g. LangChain, Google ADK runners, OpenAI Assistants). When the framework exposes a plan or workflow, Tygent parses it into the structured graph[^dag] and schedules it using the built-in executor.

### 3. Build the structured graph directly

```python
import asyncio
from tygent import DAG, ToolNode, Scheduler

dag = DAG("demo")
dag.add_node(ToolNode("search", lambda inputs: {"hits": ["url"]}))
dag.add_node(ToolNode("summarize", lambda inputs: f"Summary of {inputs['search']['hits']}"))
dag.add_edge("search", "summarize")

scheduler = Scheduler(dag)
result = asyncio.run(scheduler.execute({"query": "latest research"}))
print(result["results"]["summarize"])  # -> "Summary of ['url']"
```

The scheduler exposes token budgeting, request throttling, audit hooks, and critical path prioritisation through `Scheduler.configure`.

### 4. Adaptive execution

```python
import asyncio
from tygent import AdaptiveExecutor, create_fallback_rule
from tygent import ToolNode, DAG

# Base graph
base = DAG("workflow")
base.add_node(ToolNode("primary", lambda inputs: 1 / inputs.get("divisor", 1)))

executor = AdaptiveExecutor(
    base,
    rewrite_rules=[
        create_fallback_rule(
            error_condition=lambda state: "error" in state.get("results", {}).get("primary", {}),
            fallback_node_creator=lambda dag, state: ToolNode("fallback", lambda _inputs: 1),
            rule_name="fallback_on_error",
        )
    ],
)


async def main() -> None:
    outputs = await executor.execute({"divisor": 0})
    print(outputs["results"].keys())


asyncio.run(main())
```

Rewrite rules receive intermediate state and can inject new nodes or branches before the scheduler re-runs the structured graph[^dag].

### 5. Coordinate multiple agents

```python
import asyncio
from tygent import MultiAgentManager

manager = MultiAgentManager("support")


class Analyzer:
    async def execute(self, inputs):
        return {"keywords": inputs["question"].split()}


class Retrieval:
    async def execute(self, inputs):
        return {"docs": ["reset-guide.md"]}


manager.add_agent("analyzer", Analyzer())
manager.add_agent("retrieval", Retrieval())


async def main() -> None:
    result = await manager.execute({"question": "How do I reset my password?"})
    print(result)


asyncio.run(main())
```

### Cyclic plans, interactive nodes, and session state

Execution graphs no longer have to be acyclic. You can describe strongly connected components in plan dictionaries by adding a `loop` (or `cycle`) block to each step, optionally including a termination policy. The parser attaches the specification to the DAG metadata and the scheduler translates it into an appropriate `TerminationPolicy`:

```python
plan = {
    "name": "customer_follow_up",
    "steps": [
        {
            "name": "gather_context",
            "func": fetch_context,
            "loop": {
                "group": "follow_up",
                "termination": {"type": "fixed_point", "max_iterations": 5},
            },
            "interactive": True,  # surfaces prompts via scheduler hooks
            "session": {"persist": True},  # opt-in to persistent node state
        },
        {
            "name": "draft_reply",
            "func": render_reply,
            "dependencies": ["gather_context"],
            "loop": {"group": "follow_up"},
        },
    ],
}
```

- **Cyclic subgraphs** – every step in the same `loop.group` participates in the strongly connected component. The scheduler replays the component until the policy (single pass or fixed-point convergence) stops it. You can register additional policies programmatically with `scheduler.register_termination_policy([...], policy)`.
- **Interactive nodes** – setting `interactive: true` adds metadata consumed by higher-level runtimes so nodes can pause execution, await user input, or surface incremental updates via hooks.
- **Persistent session state** – use the `session` block to mark nodes whose results should be stored in the `SessionStore`. Nodes receive a `NodeContext` during execution and can call `context.load_state()` / `context.save_state()` to read or write cross-run state. The default `InMemorySessionStore` keeps data in-process, and you can inject your own store via `Scheduler(session_store=...)`.

These additions are backwards compatible: DAGs without cycles continue to execute with a single pass, and nodes that ignore the lifecycle hooks still work as before.

### LangGraph document-generation FSD example

Tygent plugs into LangGraph-style workflows without giving up lifecycle features. The snippet below sketches a functional-specification (FSD) writer that iterates between drafting and review steps until the reviewers approve the document, while caching progress across sessions:

```python
import asyncio
from langgraph.graph import StateGraph  # pseudo LangGraph API for illustration
from tygent import Scheduler, parse_plan

langgraph_plan = {
    "name": "fsd_writer",
    "steps": [
        {
            "name": "ingest_requirements",
            "func": lambda inputs: {"requirements": inputs["brief"]},
            "metadata": {"tags": ["ingest"]},
            "session": {"persist": True},
        },
        {
            "name": "draft_spec",
            "func": lambda inputs: {"draft": render_markdown(inputs)},
            "dependencies": ["ingest_requirements"],
            "loop": {
                "group": "fsd_iteration",
                "termination": {"type": "fixed_point", "max_iterations": 4},
            },
            "interactive": True,
            "session": {"persist": True},
        },
        {
            "name": "collect_feedback",
            "func": request_signoff,  # async function that prompts reviewers
            "dependencies": ["draft_spec"],
            "loop": {"group": "fsd_iteration"},
            "interactive": True,
        },
        {
            "name": "apply_feedback",
            "func": lambda inputs: merge_feedback(inputs["draft_spec"], inputs["collect_feedback"]),
            "dependencies": ["draft_spec", "collect_feedback"],
            "loop": {"group": "fsd_iteration"},
        },
        {
            "name": "publish_spec",
            "func": lambda inputs: store_spec(inputs["draft_spec"]),
            "dependencies": ["apply_feedback"],
            "critical": True,
        },
    ],
}

dag, critical = parse_plan(langgraph_plan)
scheduler = Scheduler(dag)
scheduler.priority_nodes = critical

async def main() -> None:
    outputs = await scheduler.execute({"brief": open("./brief.md").read()})
    print("Final draft:", outputs["results"]["publish_spec"])

asyncio.run(main())
```

**Why Tygent?**
- **Cyclic subgraphs without custom plumbing** – the review loop is encoded declaratively and the scheduler enforces the fixed-point policy, so you retain LangGraph’s expressiveness without manual orchestration.
- **Interactive checkpoints** – reviewer prompts surface through node hooks, letting humans approve or edit drafts mid-run while the scheduler resumes automatically.
- **Persistent session state** – drafts survive retries and subsequent sessions via the pluggable `SessionStore`, meaning teammates can pause/resume the FSD workflow without losing context.
- **Critical-path prioritisation & audit trail** – the publish step is marked critical, ensuring it receives resources first, while audit hooks capture every iteration for compliance reporting.

Run `examples/langgraph_fsd_example.py` to see the loop, reviewer checkpoints, and persisted state in action.

## Planner adapters

Tygent can ingest planning payloads from popular IDE assistants out of the box.
The adapters in `tygent.integrations.{claude_code, claude_code_cli, gemini_cli, openai_codex}`
turn the structures that Claude Code, Claude Code CLI, Gemini CLI, and OpenAI Codex emit into
`ServicePlan` objects that the scheduler can execute immediately.

```python
import asyncio

from tygent.integrations.claude_code import ClaudeCodePlanAdapter
from tygent.service_bridge import execute_service_plan

adapter = ClaudeCodePlanAdapter(payload)  # payload comes from Claude Code
service_plan = adapter.to_service_plan()
result = asyncio.run(execute_service_plan(service_plan, context_inputs))
```

If you call `tygent.install()` (or let the VS Code / Cursor extensions insert the
bootstrap snippet) the adapters patch their respective clients automatically, so
new planning payloads arrive in Tygent's structured format without extra glue
code.

## Service bridge and SaaS example

The Python repository bundles a mini SaaS-style planner under `tyapi/`:

- `ServicePlanBuilder` converts JSON specs (e.g. from the service) into executable plans by templating prompts, tagging critical nodes, and wiring redundancy hints.
- `execute_service_plan` prefetches referenced links (via `prefetch_many`) and executes the resulting structured graph[^dag] with optional parallelism limits.
- `tyapi.service.cli` provides commands to register accounts, issue API keys, configure ingestors, and run an aiohttp server that exposes `/v1/plan/convert` and `/v1/plan/benchmark`.

Run the service locally:

```bash
# Install in editable mode for development
pip install -e .[dev]

# Register an account and start the server
python -m tyapi.service.cli register --name "Acme" --email "ops@example.com"
python -m tyapi.service.cli serve --port 8080
```

The accompanying web UI (served from `tyapi/frontend/`) lets you paste framework-specific plans, choose redundancy settings, and compare sequential vs accelerated execution latencies.

## Examples & integrations

A collection of runnable samples lives in `examples/`:

- `advanced_python_example.py` – end-to-end structured graph[^dag] creation and execution
- `dynamic_dag_example.py` – AdaptiveExecutor rewrite rules in action on the structured graph[^dag]
- `langchain_integration.py` – working with popular agent frameworks
- `crewai_market_analysis.py`, `google_adk_market_analysis.py` – integration-specific accelerators

Call `tygent.install()` to load integration patches (Anthropic, Google AI, HuggingFace, Microsoft AI, Salesforce) before instantiating their SDK clients.

## Editor extensions

Tygent ships helper extensions for embedding the structured planner inside popular IDEs:

- **VS Code** (`vscode-extension/`) – the *Tygent: Enable Agent* command inserts `tygent.install()` and required imports into the active Python file, making it easy to convert agents in place.
- **Cursor** (`cursor-extension/`) – mirrors the VS Code command with a Cursor-specific *Tygent: Enable Agent (Cursor)* action so Cursor users can patch working files without leaving the editor.

Build either extension with `npm run compile` inside the respective folder, then use VS Code’s `Extension Development Host` or Cursor’s extension loader to test install the generated package.

## Testing

```bash
pip install -e .[dev]
pytest tests -q
```

Targeted suites exist for the tyapi service (`pytest tyapi/tests -q`) and core structured graph behaviour[^dag] (`pytest tests/test_dag.py`). The repository uses `pytest-asyncio` for async flows; see `pytest.ini` for configuration.

## Project layout

```
tygent/
│   accelerate.py      # drop-in wrappers and framework adapters
│   scheduler.py       # execution engine with hooks & budgets
│   adaptive_executor.py
│   multi_agent.py
│   service_bridge.py
│   integrations/      # opt-in SDK patches
└── tyapi/             # SaaS planner service and CLI
```

Additional tooling (editor extensions, docs) lives under `cursor-extension/`, `vscode-extension/`, and `docs/`.

[^dag]: Tygent materialises plans as typed directed acyclic graphs (DAGs) so dependencies, context-prefetch hints, and critical paths stay explicit for the execution engine and Recontext-compatible context fabric.

---
**Transform your agents. Accelerate your AI.**
Need help? Open a GitHub issue or reach out at support@tygent.ai.
