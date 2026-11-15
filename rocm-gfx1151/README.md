# Pixal3D on gfx1151 (AMD Strix Halo / Radeon 8060S)

Notes and patches for running Pixal3D single-image → GLB on a Ryzen AI Max+ 395 (gfx1151) with ROCm 7.13.
On top of yuripourre/Pixal3D-ROCm. Turtle sample: about 5.5 min end to end, peak GPU memory about 20 GB.

## Environment
- Python 3.12 venv; torch 2.11.0+rocm7.13.0 and torchvision from `https://repo.amd.com/rocm/whl/gfx1151/`,
  plus `rocm[devel]==7.13.0` (gives hipcc inside the venv). `requirements-local.txt` relaxes the trimesh pin for MoGe.
- `env.sh` holds every runtime variable. Source it before you build or run. Two of them matter most:
  - `PYTORCH_HIP_ALLOC_CONF=expandable_segments:False` **must stay off**. With it on, stage 1 returns an empty
    sparse structure: the run looks fast, then crashes later.
  - `TORCHINDUCTOR_COMPILE_THREADS=1`, because the inductor worker pool deadlocks.

## Extensions (built for gfx1151)
| ext | notes |
|---|---|
| FlexGEMM | builds clean |
| CuMesh | `patches/cumesh-rocm.diff`: `Vec3f()` gets `__host__ __device__`, a rocprim tuple decomposer, and no `--extended-lambda` for HIP. Build with `BUILD_TARGET=rocm`. |
| o-voxel (TRELLIS.2) | run `git submodule update --init o-voxel` first (it brings Eigen); install with `--no-deps` so the unpatched CuMesh isn't rebuilt |
| nvdiffrast v0.4.0 | `patches/nvdiffrast-rocm.diff`: force-included hip_compat.h, 52 PTX helpers rewritten. texture/interpolate/antialias work. `rasterize` faults above ~200 px, so it is replaced (below). |
| natten 0.21 | install without libnatten: `HIP_VISIBLE_DEVICES=-1 uv pip install --no-build-isolation --no-deps natten-0.21.0` |

## Runtime shims (auto-installed on ROCm from `pixal3d/__init__.py`)
- `pixal3d/utils/torch_rasterizer.py`: a pure-PyTorch rasterizer that replaces `dr.rasterize`
  (matches nvdiffrast on 99.6–100% of triangle ids; 1M tris → 4096² in 0.23 s).
- `pixal3d/utils/rocm_compat.py`:
  - NAF's `na2d` call goes to an exact, row-chunked PyTorch neighbourhood attention (dilation included).
  - The natten flex compile gates are patched.
  - `PIXAL_GPU_MEM_GB` caps the process's GPU memory (default 40; 0 turns it off).

## Run
```bash
source rocm-gfx1151/env.sh   # P3D defaults to the folder above this repo (holding .venv); override if needed
PIXAL_TIMING=1 python inference.py --image assets/images/0_img.png --output out.glb --smart_vram
```
Measured at 85 W: the run is bound by memory bandwidth, so going to 100 or 120 W makes no difference.
