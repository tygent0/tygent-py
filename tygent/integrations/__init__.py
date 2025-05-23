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

__all__ = [
    "LangChainIntegration",
    "OpenAIIntegration",
    "GoogleAIIntegration",
    "MicrosoftAIIntegration",
    "SalesforceIntegration",
    "AutoGenIntegration",
    "LangGraphIntegration",
    "CrewAIIntegration"
]