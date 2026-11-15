from . import models
from . import modules
from . import pipelines
from . import renderers
from . import representations
from . import utils

# ROCm shims (no-op on CUDA builds): the PyTorch rasterizer in place of nvdiffrast's HIP one, which faults
# above ~200 px on gfx1151 (PIXAL_TORCH_RASTER=0 disables), and natten cutlass->flex backends because
# libnatten is not built on AMD (PIXAL_ROCM_NATTEN=0 disables). See pixal3d/utils/rocm_compat.py.
from .utils import rocm_compat as _rocm_compat
_rocm_compat.install()
