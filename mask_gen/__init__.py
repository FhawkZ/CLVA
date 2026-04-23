"""CLVA mask generation utilities.

Combines SAM3 (for initial prompt-to-mask) with Cutie (for temporal mask
propagation). See :class:`MaskGenerator` for the main entry point.
"""

from .mask_generator import MaskGenerator, TargetSpec

__all__ = ["MaskGenerator", "TargetSpec"]
