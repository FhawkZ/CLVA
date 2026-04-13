"""Top-level agent abstractions for CLVA."""

from dataclasses import dataclass
from typing import Any

@dataclass
class ChunkAgent:
    """Minimal agent skeleton for chunk-based decision making."""

    name: str = "chunk_agent"
    chunk_size: int = 8

    def act_chunk(self, observation: Any) -> Any:
        """Return a chunk-level action plan from current observation."""
        raise NotImplementedError("Implement chunk planning logic in subclasses.")
