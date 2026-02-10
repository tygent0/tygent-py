"""Claude Code CLI planning integration for Tygent.

Claude Code CLI can emit planning payloads that contain task lists, dependency
edges, and optional metadata. This module adapts those payloads into
``ServicePlan`` objects so they can run through Tygent's scheduler and service
bridge runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Mapping, MutableMapping, Sequence

from ..service_bridge import ServicePlan, ServicePlanBuilder


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Iterable):
        items: List[str] = []
        for item in value:
            if isinstance(item, str):
                items.append(item)
            elif isinstance(item, Mapping) and "url" in item:
                url = item.get("url")
                if isinstance(url, str):
                    items.append(url)
        return items
    return []


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes", "critical"}
    return bool(value)


def _extract_plan(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    plan = payload.get("plan")
    if isinstance(plan, Mapping):
        return plan
    workflow = payload.get("workflow")
    if isinstance(workflow, Mapping):
        return workflow
    return payload


@dataclass
class ClaudeCodeCLIPlanAdapter:
    """Convert Claude Code CLI planning payloads into ``ServicePlan`` objects."""

    payload: Mapping[str, Any]
    builder: ServicePlanBuilder = ServicePlanBuilder()

    def to_service_plan(self) -> ServicePlan:
        plan_section = _extract_plan(self.payload)
        steps = (
            plan_section.get("tasks")
            or plan_section.get("steps")
            or plan_section.get("nodes")
            or []
        )
        if not isinstance(steps, Sequence):
            raise ValueError("Claude Code CLI payload requires a sequence of tasks")

        plan_name = (
            plan_section.get("plan_id")
            or plan_section.get("session_id")
            or plan_section.get("name")
            or self.payload.get("plan_id")
            or self.payload.get("session_id")
            or self.payload.get("name")
            or "claude_code_cli_plan"
        )
        plan_payload: MutableMapping[str, Any] = {"name": str(plan_name), "steps": []}

        for task in steps:
            if not isinstance(task, Mapping):
                raise TypeError("Each Claude Code CLI task must be a mapping")
            plan_payload["steps"].append(self._format_step(task))

        links = self._prefetch_links(plan_section)
        if links:
            plan_payload["prefetch"] = {"links": links}

        return self.builder.build(plan_payload)

    # ------------------------------------------------------------------
    def _format_step(self, task: Mapping[str, Any]) -> Mapping[str, Any]:
        name = (
            task.get("name") or task.get("id") or task.get("step") or task.get("title")
        )
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Claude Code CLI task missing 'name' or 'id'")

        prompt = (
            task.get("prompt")
            or task.get("instruction")
            or task.get("description")
            or task.get("command")
        )
        prompt_str = str(prompt) if prompt is not None else ""

        dependencies = _as_list(
            task.get("dependencies")
            or task.get("deps")
            or task.get("requires")
            or task.get("parents")
        )

        metadata = {
            **_as_mapping(task.get("metadata")),
        }
        provider = task.get("provider") or metadata.get("provider") or "anthropic"
        metadata.setdefault("provider", provider)
        metadata.setdefault("framework", "claude_code_cli")
        metadata.setdefault("prompt", prompt_str)
        metadata.setdefault("kind", task.get("kind", "llm"))
        level = task.get("level") or task.get("stage")
        if level is not None and "level" not in metadata:
            metadata["level"] = level

        tags = set(_as_list(task.get("tags")) + _as_list(metadata.get("tags")))
        tags.add("claude-code-cli")

        links = _as_list(task.get("links") or task.get("resources") or task.get("urls"))

        token_estimate = (
            task.get("token_estimate")
            or task.get("tokens")
            or metadata.get("token_estimate")
        )
        if token_estimate is not None and "token_estimate" not in metadata:
            metadata["token_estimate"] = token_estimate

        is_critical = _coerce_bool(
            task.get("critical")
            or task.get("is_critical")
            or metadata.get("is_critical")
        )

        return {
            "name": name,
            "kind": metadata.get("kind", "llm"),
            "prompt": prompt_str,
            "dependencies": dependencies,
            "metadata": metadata,
            "tags": sorted(tags),
            "links": links,
            "is_critical": is_critical,
            "token_estimate": token_estimate,
        }

    # ------------------------------------------------------------------
    def _prefetch_links(self, plan_section: Mapping[str, Any]) -> List[str]:
        links: List[str] = []
        links.extend(_as_list(self.payload.get("prefetch_links")))
        links.extend(_as_list(plan_section.get("prefetch_links")))
        prefetch = plan_section.get("prefetch")
        if isinstance(prefetch, Mapping):
            links.extend(_as_list(prefetch.get("links")))
        links.extend(_as_list(plan_section.get("links")))
        links.extend(_as_list(plan_section.get("resources")))
        links.extend(_as_list(plan_section.get("attachments")))
        links.extend(_as_list(self.payload.get("resources")))
        links.extend(_as_list(self.payload.get("attachments")))
        return list(dict.fromkeys(links))


def patch() -> None:
    """Attach ``to_tygent_service_plan`` to Claude Code CLI planners if present."""

    try:
        import claude_code_cli.planner as claude_cli_planner  # type: ignore
    except Exception:  # pragma: no cover - optional dependency absent
        return

    planner_cls = getattr(claude_cli_planner, "Planner", None)
    if planner_cls is None or hasattr(planner_cls, "to_tygent_service_plan"):
        return

    def to_tygent_service_plan(self, payload: Mapping[str, Any]) -> ServicePlan:
        adapter = ClaudeCodeCLIPlanAdapter(payload)
        return adapter.to_service_plan()

    setattr(planner_cls, "to_tygent_service_plan", to_tygent_service_plan)


__all__ = ["ClaudeCodeCLIPlanAdapter", "patch"]
