"""Policy interfaces for ChunkRL."""

from dataclasses import dataclass
from typing import Any


@dataclass
class BasePolicy:
    """Minimal chunk policy interface."""

    policy_name: str = "base_policy"

    def plan(self, encoded_obs: Any) -> Any:
        """Plan one chunk of actions from encoded observations."""
        raise NotImplementedError("Implement policy planning in subclasses.")
