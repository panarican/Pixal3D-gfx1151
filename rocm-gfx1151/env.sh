# Source before building or running Pixal3D-ROCm on a gfx1151 machine (Strix Halo / Radeon 8060S).
# P3D = the folder that holds .venv (python 3.12, torch ROCm) and this repo; override it before sourcing.
P3D=${P3D:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
export ROCM_HOME=$P3D/.venv/lib/python3.12/site-packages/_rocm_sdk_devel
export ROCM_PATH=$ROCM_HOME HIP_PATH=$ROCM_HOME
export PATH=$P3D/.venv/bin:$ROCM_HOME/bin:$PATH
export PYTORCH_ROCM_ARCH=gfx1151
export MAX_JOBS=16
# runtime (from the port's README)
export ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
export MIOPEN_FIND_MODE=FAST
export GPU_ARCHS=gfx1151
# compile in-process: the inductor worker pool deadlocked on compiled flex attention (2026-11-14)
export TORCHINDUCTOR_COMPILE_THREADS=1
# keep compiled kernels + autotune results across reboots (/tmp is RAM-backed and wiped on reboot)
export TORCHINDUCTOR_CACHE_DIR=$P3D/.cache/inductor
export TRITON_CACHE_DIR=$P3D/.cache/triton
# Allocator: expandable_segments MUST stay off on this ROCm setup (2026-11-14). With it on, stage 1 (sparse structure)
# returns an empty/near-empty grid: runs looked "10x faster" because later stages had almost nothing to do, then
# crashed (empty tensor in stage 2, or CuMesh "invalid argument" at export). Off = correct output (turtle GLB).
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
