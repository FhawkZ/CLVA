"""SAM3 + Cutie combined mask tracker.

High-level idea:
    - ``SAM3`` (image-level promptable concept segmentation) produces an
      initial binary mask for every target from either text or bounding-box
      prompts on the *first* frame.
    - ``Cutie`` (InferenceCore) memorises those masks and propagates them to
      every subsequent frame.

The :class:`MaskGenerator` class is the only public entry-point. It is
deliberately framework-agnostic: it consumes numpy / PIL / torch images and
returns ``dict[obj_id, np.ndarray]`` so it can be dropped into an offline
preprocessing script, a Gym ``ObservationWrapper`` for RL, or a notebook.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

__all__ = ["TargetSpec", "MaskGenerator"]


ImageLike = Union[np.ndarray, Image.Image, torch.Tensor]
BoxXYXY = Sequence[float]


@dataclass
class TargetSpec:
    """Specification for a single tracking target.

    At least one of ``text`` or ``box`` must be provided.

    Attributes:
        text: Natural-language concept prompt (e.g. ``"white mug"``). Fed to
            SAM3 as a text prompt.
        box: ``[x1, y1, x2, y2]`` bounding box in **absolute pixel
            coordinates**. Used as a positive visual prompt for SAM3.
        negative_boxes: Optional list of boxes that should be *excluded* from
            the concept (SAM3 supports combining positive text + negative
            boxes). Each box is xyxy in pixel coords.
        obj_id: Integer id assigned to this target inside the Cutie memory
            bank (must be >= 1). If ``None``, the generator auto-assigns
            consecutive ids starting from 1.
        select: How to handle the case where SAM3 returns multiple instances
            for a single prompt.

            - ``"top"``: keep only the highest-score instance (default).
            - ``"largest"``: keep the largest-area instance.
            - ``"all"``: OR-merge every instance whose score passes the
              threshold into a single mask for this target.

        score_threshold / mask_threshold: Per-target overrides of the
            generator-wide defaults.
    """

    text: Optional[str] = None
    box: Optional[BoxXYXY] = None
    negative_boxes: Optional[List[BoxXYXY]] = None
    obj_id: Optional[int] = None
    select: str = "top"
    score_threshold: Optional[float] = None
    mask_threshold: Optional[float] = None

    def __post_init__(self) -> None:
        if self.text is None and self.box is None:
            raise ValueError("TargetSpec requires at least one of `text` or `box`")
        if self.select not in ("top", "largest", "all"):
            raise ValueError(
                f"TargetSpec.select must be 'top', 'largest' or 'all', got {self.select!r}"
            )
        if self.obj_id is not None and self.obj_id < 1:
            raise ValueError("obj_id must be >= 1 (0 is reserved for background)")


class MaskGenerator:
    """Combined SAM3 + Cutie online mask tracker.

    Typical usage::

        mg = MaskGenerator()
        mg.set_targets(
            first_frame_rgb,
            [
                {"text": "white mug", "obj_id": 1},
                {"text": "metal bowl", "obj_id": 2},
            ],
        )
        for frame in frames[1:]:
            masks = mg.step(frame)          # {1: HxW bool, 2: HxW bool}

    The class maintains Cutie's internal memory across calls to :meth:`step`;
    call :meth:`reset` to start a brand-new tracking session, or
    :meth:`add_target` / :meth:`remove_target` to mutate the target set
    without dropping existing memory.
    """

    def __init__(
        self,
        sam3_model_path: Union[str, Path] = "/media/data/liqz/sam3",
        cutie_repo_path: Optional[Union[str, Path]] = "/media/data/liqz/Cutie",
        device: Union[str, torch.device] = "cuda",
        cutie_max_internal_size: int = 480,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        sam_dtype: Optional[torch.dtype] = None,
        use_amp: bool = True,
    ):
        """
        Args:
            sam3_model_path: Path to a local SAM3 HuggingFace checkpoint
                directory (must contain ``config.json`` + ``model.safetensors``)
                or a HF model id.
            cutie_repo_path: Path to the Cutie source repo. If ``cutie`` is
                already importable on ``sys.path`` you can pass ``None``.
            device: Torch device for both models.
            cutie_max_internal_size: Forwarded to ``InferenceCore``. Cutie
                resizes the shortest side of the frame to this value for
                speed; predictions are upsampled back to the original size.
                Pass ``-1`` to disable resizing.
            score_threshold / mask_threshold: Defaults used when post-processing
                SAM3 instance segmentation outputs. Can be overridden per
                target via :class:`TargetSpec`.
            sam_dtype: Optional dtype (e.g. ``torch.bfloat16``) for the SAM3
                weights to save VRAM. ``None`` keeps the default.
            use_amp: Wrap SAM3 and Cutie forward passes in
                ``torch.autocast("cuda")`` when the device is a CUDA device.
        """
        self.device = torch.device(device)
        self.score_threshold = float(score_threshold)
        self.mask_threshold = float(mask_threshold)
        self._use_amp = bool(use_amp) and self.device.type == "cuda"

        # --- SAM3 ---
        from transformers import Sam3Model, Sam3Processor

        self._sam_processor = Sam3Processor.from_pretrained(str(sam3_model_path))
        sam_model = Sam3Model.from_pretrained(str(sam3_model_path))
        if sam_dtype is not None:
            sam_model = sam_model.to(dtype=sam_dtype)
        self._sam_model = sam_model.to(self.device).eval()
        self._sam_dtype = next(self._sam_model.parameters()).dtype

        # --- Cutie ---
        if cutie_repo_path is not None:
            cutie_repo = str(Path(cutie_repo_path).resolve())
            if cutie_repo not in sys.path:
                sys.path.insert(0, cutie_repo)

        self._cutie_max_internal_size = int(cutie_max_internal_size)
        self._cutie_network, self._cutie_cfg = self._load_cutie_network()
        self._processor = self._make_cutie_processor()

        # per-target bookkeeping
        self._next_auto_id = 1
        self._targets: Dict[int, TargetSpec] = {}
        self._last_masks: Dict[int, np.ndarray] = {}
        self._last_idx_mask: Optional[np.ndarray] = None
        self._frame_hw: Optional[Tuple[int, int]] = None

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    @property
    def target_ids(self) -> List[int]:
        """Currently tracked object ids in registration order."""
        return list(self._targets.keys())

    @property
    def last_masks(self) -> Dict[int, np.ndarray]:
        """The most recent per-object masks, ``{obj_id: (H, W) bool ndarray}``."""
        return dict(self._last_masks)

    @property
    def last_idx_mask(self) -> Optional[np.ndarray]:
        """The most recent combined mask, ``(H, W) uint16`` (0 = background)."""
        return None if self._last_idx_mask is None else self._last_idx_mask.copy()

    def reset(self) -> None:
        """Forget every target and rebuild Cutie's memory from scratch.

        Equivalent to constructing a fresh :class:`MaskGenerator` minus the
        (expensive) weight loading.
        """
        self._processor = self._make_cutie_processor()
        self._targets.clear()
        self._last_masks.clear()
        self._last_idx_mask = None
        self._frame_hw = None
        self._next_auto_id = 1

    @torch.inference_mode()
    def set_targets(
        self,
        image: ImageLike,
        targets: Sequence[Union[TargetSpec, Dict[str, Any]]],
        *,
        reset: bool = True,
    ) -> Dict[int, np.ndarray]:
        """Register a set of targets using SAM3 on ``image``.

        By default this wipes the Cutie memory (so the same frame becomes the
        new "frame 0"). Pass ``reset=False`` to append the new targets to the
        existing tracking session -- but note that Cutie will then also keep
        propagating every previously-registered target.

        Args:
            image: The reference frame (typically the first frame of the
                video / rollout). See :func:`_to_pil` for accepted formats.
            targets: A list of :class:`TargetSpec` or plain dicts with the
                same fields.
            reset: Whether to clear the existing tracking session.

        Returns:
            ``{obj_id: HxW bool ndarray}`` with the initial mask per target.
        """
        specs = [_coerce_target(t) for t in targets]
        if not specs:
            raise ValueError("`targets` must be a non-empty sequence")

        if reset:
            self.reset()

        pil_image = _to_pil(image)
        self._frame_hw = (pil_image.height, pil_image.width)

        new_id_to_mask: Dict[int, np.ndarray] = {}
        for spec in specs:
            obj_id = self._assign_obj_id(spec)
            mask = self._sam3_mask_for_target(pil_image, spec)
            new_id_to_mask[obj_id] = mask
            self._targets[obj_id] = spec

        idx_mask = self._stack_masks(new_id_to_mask)
        image_t = self._frame_to_tensor(pil_image)

        if reset:
            objects = list(new_id_to_mask.keys())
            self._cutie_step(image_t, mask=idx_mask, objects=objects)
        else:
            # Only the newly-added ids are memorised; pre-existing ones
            # propagate from Cutie memory.
            objects = list(new_id_to_mask.keys())
            self._cutie_step(image_t, mask=idx_mask, objects=objects)

        # The ``step`` above already populates ``self._last_masks``.
        return {k: self._last_masks[k] for k in new_id_to_mask}

    @torch.inference_mode()
    def add_target(
        self,
        image: ImageLike,
        *,
        text: Optional[str] = None,
        box: Optional[BoxXYXY] = None,
        negative_boxes: Optional[List[BoxXYXY]] = None,
        obj_id: Optional[int] = None,
        select: str = "top",
        score_threshold: Optional[float] = None,
        mask_threshold: Optional[float] = None,
    ) -> Tuple[int, np.ndarray]:
        """Add a single new target mid-track.

        The existing tracked objects are preserved; SAM3 runs only on the
        current frame to produce the initial mask for this new object, which
        is then registered into Cutie's memory.

        Returns:
            ``(obj_id, mask)``.
        """
        spec = TargetSpec(
            text=text,
            box=box,
            negative_boxes=list(negative_boxes) if negative_boxes else None,
            obj_id=obj_id,
            select=select,
            score_threshold=score_threshold,
            mask_threshold=mask_threshold,
        )
        return next(iter(self.set_targets(image, [spec], reset=False).items()))

    def remove_target(self, obj_id: int) -> None:
        """Stop tracking ``obj_id`` (drops it from Cutie's memory)."""
        if obj_id not in self._targets:
            raise KeyError(f"obj_id={obj_id} is not currently tracked")
        self._processor.delete_objects([obj_id])
        self._targets.pop(obj_id)
        self._last_masks.pop(obj_id, None)

    @torch.inference_mode()
    def step(self, image: ImageLike) -> Dict[int, np.ndarray]:
        """Propagate the tracked masks to a new frame.

        Must be called after at least one successful :meth:`set_targets` /
        :meth:`add_target`. Returns the per-object masks for the new frame.
        """
        if not self._targets:
            raise RuntimeError(
                "No targets registered; call `set_targets(...)` before `step(...)`."
            )
        pil_image = _to_pil(image)
        self._frame_hw = (pil_image.height, pil_image.width)
        image_t = self._frame_to_tensor(pil_image)
        self._cutie_step(image_t, mask=None, objects=None)
        return dict(self._last_masks)

    # ------------------------------------------------------------------
    # SAM3 helpers
    # ------------------------------------------------------------------
    def _sam3_mask_for_target(self, pil_image: Image.Image, spec: TargetSpec) -> np.ndarray:
        """Run SAM3 for a single target spec and return a ``(H, W) bool`` mask."""
        H, W = pil_image.height, pil_image.width

        processor_kwargs: Dict[str, Any] = {"images": pil_image, "return_tensors": "pt"}
        if spec.text is not None:
            processor_kwargs["text"] = spec.text

        boxes: List[BoxXYXY] = []
        box_labels: List[int] = []
        if spec.box is not None:
            boxes.append(list(spec.box))
            box_labels.append(1)
        if spec.negative_boxes:
            for nb in spec.negative_boxes:
                boxes.append(list(nb))
                box_labels.append(0)
        if boxes:
            processor_kwargs["input_boxes"] = [boxes]
            processor_kwargs["input_boxes_labels"] = [box_labels]

        inputs = self._sam_processor(**processor_kwargs).to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda", dtype=self._sam_dtype)
            if self._use_amp
            else _null_ctx()
        )
        with autocast_ctx:
            outputs = self._sam_model(**inputs)

        score_th = spec.score_threshold if spec.score_threshold is not None else self.score_threshold
        mask_th = spec.mask_threshold if spec.mask_threshold is not None else self.mask_threshold

        results = self._sam_processor.post_process_instance_segmentation(
            outputs,
            threshold=score_th,
            mask_threshold=mask_th,
            target_sizes=[[H, W]],
        )[0]

        masks = results["masks"]   # (N, H, W) bool/float tensor
        scores = results["scores"] # (N,) float tensor

        if masks is None or len(masks) == 0:
            raise RuntimeError(
                f"SAM3 returned no instances for target {spec!r} (threshold={score_th})."
                " Try lowering score_threshold or giving a more specific prompt/box."
            )

        masks_np = masks.detach().cpu().numpy().astype(bool)
        scores_np = scores.detach().cpu().numpy().astype(np.float32)

        if spec.select == "top":
            keep = int(np.argmax(scores_np))
            mask = masks_np[keep]
        elif spec.select == "largest":
            areas = masks_np.reshape(masks_np.shape[0], -1).sum(axis=1)
            mask = masks_np[int(np.argmax(areas))]
        elif spec.select == "all":
            mask = np.any(masks_np, axis=0)
        else:  # pragma: no cover -- validated in TargetSpec
            raise AssertionError(spec.select)

        return mask.astype(bool)

    # ------------------------------------------------------------------
    # Cutie helpers
    # ------------------------------------------------------------------
    def _load_cutie_network(self):
        """Load the Cutie network using its default ``eval_config``."""
        # `hydra.initialize` can only be called once per process. Clear any
        # leftover global state so repeated instantiations still work.
        try:
            from hydra.core.global_hydra import GlobalHydra

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
        except Exception:
            pass

        from cutie.utils.get_default_model import get_default_model

        network = get_default_model()
        cfg = network.cfg
        # `get_default_model` internally calls `.cuda()`; move it to the
        # requested device if different.
        network = network.to(self.device).eval()
        return network, cfg

    def _make_cutie_processor(self):
        from cutie.inference.inference_core import InferenceCore

        processor = InferenceCore(self._cutie_network, cfg=self._cutie_cfg)
        processor.max_internal_size = self._cutie_max_internal_size
        return processor

    def _cutie_step(
        self,
        image_t: torch.Tensor,
        mask: Optional[np.ndarray],
        objects: Optional[List[int]],
    ) -> None:
        """Run one ``InferenceCore.step`` and update ``self._last_masks``."""
        mask_tensor = None
        if mask is not None:
            mask_tensor = torch.from_numpy(mask.astype(np.int64)).to(self.device)

        autocast_ctx = (
            torch.autocast(device_type="cuda")
            if self._use_amp
            else _null_ctx()
        )
        with autocast_ctx:
            output_prob = self._processor.step(
                image_t, mask_tensor, objects=objects, idx_mask=True
            )
            idx_mask_t = self._processor.output_prob_to_mask(output_prob)

        idx_mask_np = idx_mask_t.detach().cpu().numpy().astype(np.uint16)
        self._last_idx_mask = idx_mask_np

        per_obj: Dict[int, np.ndarray] = {}
        for obj_id in self._targets:
            per_obj[obj_id] = idx_mask_np == obj_id
        self._last_masks = per_obj

    # ------------------------------------------------------------------
    # misc helpers
    # ------------------------------------------------------------------
    def _assign_obj_id(self, spec: TargetSpec) -> int:
        if spec.obj_id is not None:
            if spec.obj_id in self._targets:
                raise ValueError(
                    f"obj_id={spec.obj_id} already tracked; remove_target first"
                )
            self._next_auto_id = max(self._next_auto_id, spec.obj_id + 1)
            return spec.obj_id
        while self._next_auto_id in self._targets:
            self._next_auto_id += 1
        oid = self._next_auto_id
        self._next_auto_id += 1
        return oid

    def _stack_masks(self, id_to_mask: Dict[int, np.ndarray]) -> np.ndarray:
        """Compose per-object binary masks into a single idx mask.

        Targets earlier in the dict have *lower* priority than later ones,
        i.e. overlapping pixels end up labelled as the *last*-added target.
        This matches how most users think about "I just added a new target,
        it should win over the background prediction".
        """
        assert self._frame_hw is not None
        H, W = self._frame_hw
        idx = np.zeros((H, W), dtype=np.int64)
        for oid, m in id_to_mask.items():
            idx[m] = oid
        return idx

    def _frame_to_tensor(self, pil_image: Image.Image) -> torch.Tensor:
        # ``np.array`` (not ``asarray``) guarantees we get a writable copy;
        # frames coming from ``cv2.VideoCapture`` are sometimes read-only and
        # would otherwise trip a ``torch.from_numpy`` warning.
        arr = np.array(pil_image.convert("RGB"), dtype=np.uint8, copy=True)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous().float().div_(255.0)
        return t.to(self.device)


# ----------------------------------------------------------------------
# module-level helpers
# ----------------------------------------------------------------------
def _coerce_target(t: Union[TargetSpec, Dict[str, Any]]) -> TargetSpec:
    if isinstance(t, TargetSpec):
        return t
    if isinstance(t, dict):
        return TargetSpec(**t)
    raise TypeError(f"Expected TargetSpec or dict, got {type(t).__name__}")


def _to_pil(image: ImageLike) -> Image.Image:
    """Convert any supported image format to a ``PIL.Image`` in RGB mode.

    Accepted formats:
        - ``PIL.Image.Image`` (any mode; converted to RGB).
        - ``np.ndarray`` of shape ``(H, W, 3)`` uint8, assumed RGB.
        - ``np.ndarray`` of shape ``(H, W, 3)`` float32/64 in [0, 1], RGB.
        - ``torch.Tensor`` of shape ``(3, H, W)`` or ``(H, W, 3)``; uint8 or
          float in [0, 1].
    """
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, torch.Tensor):
        t = image.detach().cpu()
        if t.ndim != 3:
            raise ValueError(f"Tensor image must be 3-D, got shape {tuple(t.shape)}")
        if t.shape[0] == 3 and t.shape[-1] != 3:
            t = t.permute(1, 2, 0)
        elif t.shape[-1] != 3:
            raise ValueError(
                f"Tensor image must have 3 channels; got shape {tuple(t.shape)}"
            )
        arr = t.numpy()
    elif isinstance(image, np.ndarray):
        arr = image
        if arr.ndim != 3 or arr.shape[-1] != 3:
            raise ValueError(
                f"ndarray image must be (H, W, 3); got shape {arr.shape}"
            )
    else:
        raise TypeError(
            f"Unsupported image type: {type(image).__name__}. "
            "Expected PIL.Image, np.ndarray or torch.Tensor."
        )

    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0.0, 1.0)
        arr = (arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr, mode="RGB")


class _null_ctx:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False
