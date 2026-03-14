"""TRN framework integrations.

Provides adapter classes for popular agent frameworks:
    - LangGraph: TRNMemoryNode (graph node pattern)
    - AutoGen: TRNConversableAgent (mixin for conversable agents)
    - CrewAI: TRNLongTermMemory (long-term memory interface)

All adapters use Protocol/ABC — framework packages are NOT required
at import time. Install the target framework separately.

Example (LangGraph)::

    from trimemory.integrations import TRNMemoryNode

Example (AutoGen)::

    from trimemory.integrations import TRNConversableAgent

Example (CrewAI)::

    from trimemory.integrations import TRNLongTermMemory
"""
from __future__ import annotations

from .autogen_adapter import TRNConversableAgent
from .crewai_adapter import TRNLongTermMemory
from .langgraph_adapter import TRNMemoryNode

__all__ = [
    "TRNMemoryNode",
    "TRNConversableAgent",
    "TRNLongTermMemory",
]
