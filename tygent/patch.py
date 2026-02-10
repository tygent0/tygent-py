"""Central patch installer for Tygent integrations."""

from importlib import import_module
from typing import List, Optional


def install(modules: Optional[List[str]] = None) -> None:
    """Install monkey patches for supported integrations."""
    default = [
        "tygent.integrations.google_ai",
        "tygent.integrations.anthropic",
        "tygent.integrations.huggingface",
        "tygent.integrations.microsoft_ai",
        "tygent.integrations.salesforce",
        "tygent.integrations.claude_code",
        "tygent.integrations.claude_code_cli",
        "tygent.integrations.gemini_cli",
        "tygent.integrations.openai_codex",
    ]
    for mod_name in modules or default:
        try:
            mod = import_module(mod_name)
        except Exception:
            continue
        patch = getattr(mod, "patch", None)
        if callable(patch):
            try:
                patch()
            except Exception:
                continue
