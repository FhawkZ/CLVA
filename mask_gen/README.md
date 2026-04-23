# `mask_gen` — SAM3 + Cutie 在线掩码追踪

把 **SAM3**（开放词表概念分割，做"首帧提示 → 掩码"）和 **Cutie**（视频对象分割，做"逐帧传播"）封装成一个对外的 `MaskGenerator` 类。

- 首帧通过 `set_targets(image, [{"text": "white mug"}, {"box": [x1,y1,x2,y2]}])` 指定目标，类内部会调 SAM3 得到初始掩码。
- 之后每到一帧仿真/视频图像，调 `mg.step(image)`，类内部用 Cutie 基于记忆做传播，返回每个目标的二值掩码。
- 中途可以 `add_target` / `remove_target` / `reset`，Cutie 的内存管理按它自己的规矩自动处理。

## 依赖

- `transformers >= 5.5`（提供 `Sam3Model` / `Sam3Processor`；在 `mlspaces` conda env 里已装）
- `cutie`（直接用 `/media/data/liqz/Cutie/` 源码，通过 `sys.path` 注入，不需要 `pip install`）
- SAM3 权重：`/media/data/liqz/sam3/`（本地 HF checkpoint）
- Cutie 权重：`/media/data/liqz/Cutie/weights/cutie-base-mega.pth`（首次调用 `get_default_model()` 会自动检查/下载）
- GPU：推荐 >=12 GB 显存

## 快速示例

```python
import cv2
from mask_gen import MaskGenerator, TargetSpec

mg = MaskGenerator(
    sam3_model_path="/media/data/liqz/sam3",
    cutie_repo_path="/media/data/liqz/Cutie",
    cutie_max_internal_size=480,
)

cap = cv2.VideoCapture("episode_00000000_exo_camera_1_batch_1_of_1.mp4")
ok, frame = cap.read()
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # 类要求输入是 RGB

mg.set_targets(frame, [
    {"text": "white mug",                      "obj_id": 1},
    {"text": "round metallic bowl",            "obj_id": 2},
    TargetSpec(box=[770, 60, 900, 230], obj_id=3, select="largest"),  # gripper by bbox
])

while True:
    ok, frame = cap.read()
    if not ok:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    masks = mg.step(frame)         # {1: (H,W) bool, 2: (H,W) bool, 3: (H,W) bool}
    idx   = mg.last_idx_mask       # (H,W) uint16，背景=0
```

## 类接口一览

| 方法 | 说明 |
| --- | --- |
| `__init__(...)` | 构造时加载 SAM3 + Cutie，一次即可 |
| `set_targets(image, targets, *, reset=True)` | 重置并在当前帧用 SAM3 定位全部目标；返回 `{obj_id: mask}` |
| `add_target(image, *, text=None, box=None, ...)` | 保留已有追踪，仅在当前帧新增一个目标 |
| `remove_target(obj_id)` | 从 Cutie 内存中剔除一个对象 |
| `step(image)` | 把当前帧送给 Cutie 做传播，返回 `{obj_id: mask}` |
| `reset()` | 清空所有目标与 Cutie 内存 |
| `target_ids` | 当前追踪的 id 列表 |
| `last_masks` / `last_idx_mask` | 最近一次的 per-object / 合并 idx 掩码 |

### `TargetSpec` 字段

| 字段 | 说明 |
| --- | --- |
| `text` | 自然语言 prompt（SAM3 PCS）|
| `box` | 正向边界框 `[x1,y1,x2,y2]`，像素坐标 |
| `negative_boxes` | 负向框列表，用于排除干扰区域 |
| `obj_id` | 对象 id（≥1），省略即自动分配 |
| `select` | `"top"` / `"largest"` / `"all"`——SAM3 返回多个实例时的选择策略 |
| `score_threshold` / `mask_threshold` | 覆盖全局默认 |

## 输入图像格式

`set_targets` / `step` 都接受以下任一形式（要求 **RGB** 通道顺序）：

- `PIL.Image.Image`
- `np.ndarray`，`(H, W, 3)`，`uint8` 或 `float32 ∈ [0,1]`
- `torch.Tensor`，`(3, H, W)` 或 `(H, W, 3)`，`uint8` 或 `float ∈ [0,1]`

> OpenCV 默认读出来是 BGR，记得先 `cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`。

## 性能（RTX 5090, `max_internal_size=480`, 2 目标）

- `__init__`（含两次模型加载）：≈25 s（一次性）
- `set_targets`：≈2 s（SAM3 是主要开销）
- `step`（Cutie 传播）：约 35–50 ms/帧

因此 SAM3 只应在"切换目标"时调用；**每步仿真请只调 `step`**。

## 给 CLVA 三阶段的用法建议

- **Stage 1 / IL、Stage 2 / Dreamer（离线）**：写一个脚本遍历 molmospaces 生成的视频，对每个 episode `set_targets` 一次，然后 `step` 跑全序列，把 mask 序列存到数据集目录（建议 `.npz` 或 palette PNG）。后续训练的 dataloader 就地读 mask 即可。
- **Stage 3 / RL（在线）**：把同一个 `MaskGenerator` 实例包成 `gym.ObservationWrapper`——`reset()` 调 `set_targets`，`step()` 调 `self.mg.step(frame)`。这样训练-推理阶段使用的感知链路完全一致，避免分布漂移。
