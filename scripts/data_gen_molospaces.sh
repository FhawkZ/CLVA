#!/usr/bin/env bash
set -euo pipefail

# 0) 进入项目
cd /media/data/liqz/CLVA/thirdpart/molmospaces

# 1) 环境变量
export MPLCONFIGDIR=/media/data/liqz/.cache/matplotlib
export NLTK_DATA=/media/data/liqz/nltk
export MLSPACES_ASSETS_DIR=/media/data/liqz/molmospaces_assets
export PYTHONPATH=/media/data/liqz/CLVA/thirdpart/molmospaces
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 限制 BLAS/OMP 线程数，避免 XLA 编译与 CPU 仿真互相抢核
export OMP_NUM_THREADS=16
export MKL_NUM_THREADS=16

# JAX 编译缓存：首次编译仍慢，之后同配置秒级加载
export JAX_COMPILATION_CACHE_DIR=/media/data/liqz/.jax_cache
export JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES=0
export JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=1
mkdir -p "$JAX_COMPILATION_CACHE_DIR"

# 2) 日志同时输出到终端和文件
LOG_DIR=/media/data/liqz/.logs
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/data_gen_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "[INFO] 日志文件: $LOG_FILE"

# 3) 用 conda run 避免 "conda init" 依赖；检查 mujoco 是否就绪
if ! conda run -n mlspaces python -c "import mujoco" >/dev/null 2>&1; then
  echo "[ERROR] mlspaces 环境里缺少 mujoco。先执行："
  echo "  cd /media/data/liqz/CLVA/thirdpart/molmospaces && conda run -n mlspaces pip install -e '.[mujoco]'"
  exit 1
fi

# 4) 单目标 + 指定容器，跨 house 补采直到达到目标
HOUSE_CANDIDATES=(1 2 3 4 5 6 7 8 9 10)
TARGET_OBJECT_TYPE="mug"
TARGET_RECEPTACLE_TYPE="bowl"
TARGET_SUCCESS_EPISODES=100
SAMPLES_PER_ROUND=10
MAX_ROUNDS=40
SEED_BASE=7
RUN_PREFIX_BASE="pick_and_place_${TARGET_OBJECT_TYPE}_to_${TARGET_RECEPTACLE_TYPE}"

echo "[INFO] house candidates=${HOUSE_CANDIDATES[*]}"
echo "[INFO] target_object=${TARGET_OBJECT_TYPE}"
echo "[INFO] target_receptacle=${TARGET_RECEPTACLE_TYPE}"
echo "[INFO] target_success_episodes=${TARGET_SUCCESS_EPISODES}"
echo "[INFO] samples_per_round=${SAMPLES_PER_ROUND}, max_rounds=${MAX_ROUNDS}"

count_collected_episodes() {
  local run_prefix="$1"
  local out
  out="$(
    conda run --no-capture-output -n mlspaces python - "$MLSPACES_ASSETS_DIR" "$run_prefix" <<'PY'
import sys
from pathlib import Path
import h5py

assets_dir = Path(sys.argv[1])
run_prefix = sys.argv[2]
root = assets_dir / "datagen" / "pick_and_place_planner_v1"
total = 0

if root.exists():
    for run_dir in root.iterdir():
        if not run_dir.is_dir() or not run_dir.name.startswith(run_prefix):
            continue
        for h5_path in run_dir.glob("house_*/trajectories*.h5"):
            try:
                with h5py.File(h5_path, "r") as f:
                    total += sum(1 for k in f.keys() if k.startswith("traj_"))
            except OSError:
                pass

print(total)
PY
  )" || true
  out="$(printf '%s\n' "$out" | awk '/^[0-9]+$/ {v=$0} END {print v}')"
  if [ -z "$out" ]; then
    out=0
  fi
  echo "$out"
}

round_idx=1
while true; do
  collected="$(count_collected_episodes "${RUN_PREFIX_BASE}")"
  if [ "${collected}" -ge "${TARGET_SUCCESS_EPISODES}" ]; then
    echo "[INFO] 达成目标: collected=${collected} >= ${TARGET_SUCCESS_EPISODES}"
    break
  fi

  if [ "${round_idx}" -gt "${MAX_ROUNDS}" ]; then
    echo "[WARN] 达到最大轮数仍未达标: collected=${collected}, target=${TARGET_SUCCESS_EPISODES}"
    break
  fi

  house_idx=$(( (round_idx - 1) % ${#HOUSE_CANDIDATES[@]} ))
  house_id="${HOUSE_CANDIDATES[$house_idx]}"
  run_prefix="${RUN_PREFIX_BASE}_h${house_id}_r${round_idx}"
  round_seed=$((SEED_BASE + round_idx - 1))
  echo "[INFO] ===== round ${round_idx} | house=${house_id} | collected=${collected}/${TARGET_SUCCESS_EPISODES} ====="

  conda run --no-capture-output -n mlspaces python -u scripts/datagen/run_pipeline.py \
    --task_type pick_and_place \
    --policy planner \
    --robot droid \
    --scene_dataset ithor \
    --data_split train \
    --house_inds "${house_id}" \
    --target_types "${TARGET_OBJECT_TYPE}" \
    --place_receptacle_types "${TARGET_RECEPTACLE_TYPE}" \
    --samples_per_house "${SAMPLES_PER_ROUND}" \
    --filter_for_successful_trajectories \
    --seed "${round_seed}" \
    --run_name_prefix "${run_prefix}"

  round_idx=$((round_idx + 1))
done

final_collected="$(count_collected_episodes "${RUN_PREFIX_BASE}")"
echo "[INFO] final collected episodes=${final_collected}"