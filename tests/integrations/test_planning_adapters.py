import asyncio

from tygent.integrations.claude_code import ClaudeCodePlanAdapter
from tygent.integrations.claude_code_cli import ClaudeCodeCLIPlanAdapter
from tygent.integrations.gemini_cli import GeminiCLIPlanAdapter
from tygent.integrations.openai_codex import OpenAICodexPlanAdapter
from tygent.service_bridge import DEFAULT_LLM_RUNTIME, execute_service_plan


async def _mock_runtime(prompt, metadata, inputs, *, sink):
    sink.append({"prompt": prompt, "metadata": dict(metadata), "inputs": inputs})
    return {"text": prompt.upper(), "tags": metadata.get("tags", [])}


def test_claude_code_adapter_executes_plan():
    calls = []

    DEFAULT_LLM_RUNTIME.register(
        "claude-code-runtime",
        lambda prompt, metadata, inputs: _mock_runtime(
            prompt, metadata, inputs, sink=calls
        ),
    )

    payload = {
        "plan_id": "claude-session",
        "tasks": [
            {
                "id": "plan_outline",
                "prompt": "Outline fixes for {topic}",
                "metadata": {
                    "provider": "claude-code-runtime",
                    "is_critical": True,
                    "tags": ["outline"],
                },
                "links": ["https://claude.docs"],
            },
            {
                "id": "apply_patch",
                "instruction": "Implement based on {plan_outline[result][text]}",
                "deps": ["plan_outline"],
                "provider": "claude-code-runtime",
                "tokens": 128,
            },
        ],
        "resources": ["https://claude.docs"],
    }

    adapter = ClaudeCodePlanAdapter(payload)
    service_plan = adapter.to_service_plan()

    assert service_plan.prefetch_links == ["https://claude.docs"]
    steps = service_plan.plan["steps"]
    assert steps[0]["metadata"]["framework"] == "claude_code"
    assert steps[0]["metadata"]["provider"] == "claude-code-runtime"
    assert steps[1]["token_cost"] == 128

    result = asyncio.run(execute_service_plan(service_plan, {"topic": "vector search"}))

    assert {"plan_outline", "apply_patch"}.issubset(result["results"].keys())
    assert calls[0]["inputs"]["prefetch"]["https://claude.docs"] == "prefetched"
    assert len(calls) == 2
    assert calls[0]["metadata"]["provider"] == "claude-code-runtime"
    assert calls[1]["metadata"]["provider"] == "claude-code-runtime"


def test_gemini_cli_adapter_executes_plan():
    calls = []

    DEFAULT_LLM_RUNTIME.register(
        "gemini-cli-runtime",
        lambda prompt, metadata, inputs: _mock_runtime(
            prompt, metadata, inputs, sink=calls
        ),
    )

    payload = {
        "plan": {
            "name": "gemini-cli-session",
            "steps": [
                {
                    "name": "collect",
                    "instruction": "Collect articles about {topic}",
                    "provider": "gemini-cli-runtime",
                    "links": ["https://ai.google.dev"],
                    "is_critical": True,
                },
                {
                    "name": "summarise",
                    "prompt": "Summarise {collect[result][text]}",
                    "deps": ["collect"],
                    "metadata": {
                        "provider": "gemini-cli-runtime",
                        "token_estimate": 64,
                    },
                },
            ],
            "prefetch": {"links": ["https://ai.google.dev"]},
        }
    }

    adapter = GeminiCLIPlanAdapter(payload)
    service_plan = adapter.to_service_plan()

    assert service_plan.prefetch_links == ["https://ai.google.dev"]
    steps = service_plan.plan["steps"]
    assert steps[1]["metadata"]["framework"] == "gemini_cli"
    assert steps[1]["metadata"]["provider"] == "gemini-cli-runtime"
    assert steps[1]["token_cost"] == 64

    result = asyncio.run(execute_service_plan(service_plan, {"topic": "rust async"}))

    assert {"collect", "summarise"}.issubset(result["results"])  # keys present
    assert len(calls) == 2
    assert calls[0]["metadata"]["provider"] == "gemini-cli-runtime"
    assert calls[1]["inputs"]["collect"]["result"]["text"].startswith("COLLECT")


def test_claude_code_cli_adapter_executes_plan():
    calls = []

    DEFAULT_LLM_RUNTIME.register(
        "claude-code-cli-runtime",
        lambda prompt, metadata, inputs: _mock_runtime(
            prompt, metadata, inputs, sink=calls
        ),
    )

    payload = {
        "plan": {
            "session_id": "claude-cli-session",
            "tasks": [
                {
                    "id": "draft",
                    "instruction": "Draft changes for {topic}",
                    "metadata": {
                        "provider": "claude-code-cli-runtime",
                        "is_critical": "true",
                    },
                    "links": ["https://cli.docs"],
                },
                {
                    "id": "apply",
                    "instruction": "Apply using {draft[result][text]}",
                    "requires": ["draft"],
                    "provider": "claude-code-cli-runtime",
                    "token_estimate": 40,
                },
            ],
            "prefetch": {"links": ["https://cli.docs"]},
        }
    }

    adapter = ClaudeCodeCLIPlanAdapter(payload)
    service_plan = adapter.to_service_plan()

    assert service_plan.prefetch_links == ["https://cli.docs"]
    steps = service_plan.plan["steps"]
    assert steps[0]["metadata"]["framework"] == "claude_code_cli"
    assert steps[0]["metadata"]["provider"] == "claude-code-cli-runtime"
    assert steps[0]["critical"] is True
    assert steps[1]["token_cost"] == 40

    result = asyncio.run(execute_service_plan(service_plan, {"topic": "build cache"}))

    assert {"draft", "apply"}.issubset(result["results"].keys())
    assert len(calls) == 2
    assert calls[0]["inputs"]["prefetch"]["https://cli.docs"] == "prefetched"
    assert calls[1]["inputs"]["draft"]["result"]["text"].startswith("DRAFT")


def test_openai_codex_adapter_executes_plan():
    calls = []

    DEFAULT_LLM_RUNTIME.register(
        "openai-codex-runtime",
        lambda prompt, metadata, inputs: _mock_runtime(
            prompt, metadata, inputs, sink=calls
        ),
    )

    payload = {
        "workflow": {
            "name": "codex-plan",
            "nodes": [
                {
                    "id": "draft",
                    "prompt": "Generate code for {task}",
                    "metadata": {
                        "provider": "openai-codex-runtime",
                        "tags": ["draft"],
                        "is_critical": True,
                        "token_estimate": 32,
                    },
                    "links": ["https://platform.openai.com/docs"],
                },
                {
                    "id": "review",
                    "instruction": "Review {draft[result][text]}",
                    "dependencies": ["draft"],
                    "provider": "openai-codex-runtime",
                },
            ],
            "prefetch": {"links": ["https://platform.openai.com/docs"]},
        }
    }

    adapter = OpenAICodexPlanAdapter(payload)
    service_plan = adapter.to_service_plan()

    assert service_plan.prefetch_links == ["https://platform.openai.com/docs"]
    steps = service_plan.plan["steps"]
    assert steps[0]["metadata"]["framework"] == "openai_codex"
    assert steps[0]["metadata"]["provider"] == "openai-codex-runtime"
    assert steps[0]["token_cost"] == 32

    result = asyncio.run(execute_service_plan(service_plan, {"task": "write pytest"}))

    assert {"draft", "review"}.issubset(result["results"].keys())
    assert len(calls) == 2
    assert calls[1]["inputs"]["draft"]["result"]["text"].startswith("GENERATE")
    assert calls[0]["metadata"]["provider"] == "openai-codex-runtime"
