"""
Tygent Integrations

This package provides integrations with popular LLM frameworks and platforms.
"""

# Import integrations as they become available
try:
    from .langchain import LangChainIntegration
except ImportError:
    pass

try:
    from .openai import OpenAIIntegration
except ImportError:
    pass

try:
    from .google_ai import GoogleAIIntegration
except ImportError:
    pass

try:
    from .google_adk import GoogleADKIntegration
except ImportError:
    pass

try:
    from .microsoft_ai import MicrosoftAIIntegration
except ImportError:
    pass

try:
    from .salesforce import SalesforceIntegration
except ImportError:
    pass

try:
    from .autogen import AutoGenIntegration
except ImportError:
    pass

try:
    from .langgraph import LangGraphIntegration
except ImportError:
    pass

try:
    from .crewai import CrewAIIntegration
except ImportError:
    pass

try:
    from .anthropic import AnthropicIntegration
except ImportError:
    pass

try:
    from .huggingface import HuggingFaceIntegration
except ImportError:
    pass

try:
    from .claude_code import ClaudeCodePlanAdapter
except ImportError:
    pass

try:
    from .claude_code_cli import ClaudeCodeCLIPlanAdapter
except ImportError:
    pass

try:
    from .gemini_cli import GeminiCLIPlanAdapter
except ImportError:
    pass

try:
    from .openai_codex import OpenAICodexPlanAdapter
except ImportError:
    pass

try:
    from .langsmith import LangSmithTracker
except ImportError:
    pass

__all__ = [
    "LangChainIntegration",
    "OpenAIIntegration",
    "GoogleAIIntegration",
    "GoogleADKIntegration",
    "MicrosoftAIIntegration",
    "SalesforceIntegration",
    "AutoGenIntegration",
    "LangGraphIntegration",
    "CrewAIIntegration",
    "AnthropicIntegration",
    "HuggingFaceIntegration",
    "ClaudeCodePlanAdapter",
    "ClaudeCodeCLIPlanAdapter",
    "GeminiCLIPlanAdapter",
    "OpenAICodexPlanAdapter",
    "LangSmithTracker",
]
