"""
ROCm compatibility shims that must be in place before the models load (installed from pixal3d/__init__.py).

natten: on AMD, natten is installed without libnatten (its CUDA/CUTLASS kernels), so only the PyTorch
flex-attention backends work. NAF (torch.hub valeoai/NAF) hard-codes backend="cutlass-fna"; this maps any
cutlass/hopper/blackwell backend request to the matching flex backend. Verified against a brute-force
neighbourhood attention: max abs diff 2.4e-7. Different QK/V head dims are
handled by running V in QK-sized slices. NAF does `from natten import na2d` when it is loaded, which
is after this runs, so it picks up the wrapped function.
"""
from __future__ import annotations

import functools
import os

import torch

_NVIDIA_ONLY = ("cutlass-", "hopper-", "blackwell-")


def _flex_backend(backend):
    if isinstance(backend, str) and backend.startswith(_NVIDIA_ONLY):
        return "flex-" + backend.split("-", 1)[1]          # cutlass-fna -> flex-fna, cutlass-fmha -> flex-fmha
    return backend


def _local_na2d(q, k, v, kernel_size, scale=None, rows_per_chunk=None):
    """Exact 2-D neighbourhood attention (NATTEN semantics: K x K window, shifted inward at the borders) in
    plain PyTorch, a few rows at a time. No block mask, so it scales linearly with image size on ROCm, where
    flex's mask building dominates. q/k: (B, H, W, heads, d); v: (B, H, W, heads, dv)."""
    B, H, W, heads, d = q.shape
    dv = v.shape[-1]
    K = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    if (not isinstance(kernel_size, int) and kernel_size[0] != kernel_size[1]) or H < K or W < K:
        raise ValueError("local_na2d: square kernel no larger than the image only")
    r = K // 2
    scale = d ** -0.5 if scale is None else scale
    dev = q.device
    si = (torch.arange(H, device=dev) - r).clamp(0, H - K)            # window start per row
    sj = (torch.arange(W, device=dev) - r).clamp(0, W - K)            # and per column
    off = torch.arange(K, device=dev)
    col = (sj[:, None] + off[None, :])                                 # (W, K)
    if rows_per_chunk is None:   # ~1 GiB of gathered K/V per chunk
        per_row = B * W * K * K * heads * (d + dv) * 4      # gathered K/V (+ fp32 math) for one row, all batches
        rows_per_chunk = max(1, int(2 ** 30 // max(per_row, 1)))
    out = torch.empty(B, H, W, heads, dv, device=dev, dtype=v.dtype)
    for i0 in range(0, H, rows_per_chunk):
        i1 = min(H, i0 + rows_per_chunk)
        rows = si[i0:i1, None] + off[None, :]                          # (h, K)
        # gather windows: (B, h, K, W, K, heads, *) -> (B, h, W, K*K, heads, *)
        kk = k[:, rows][:, :, :, col].permute(0, 1, 3, 2, 4, 5, 6).reshape(B, i1 - i0, W, K * K, heads, d)
        vv = v[:, rows][:, :, :, col].permute(0, 1, 3, 2, 4, 5, 6).reshape(B, i1 - i0, W, K * K, heads, dv)
        att = torch.einsum("bhwnd,bhwknd->bhwnk", q[:, i0:i1].float(), kk.float()) * scale
        out[:, i0:i1] = torch.einsum("bhwnk,bhwknd->bhwnd", att.softmax(-1), vv.float()).to(v.dtype)
    return out


def _wrap(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # 2-D, plain (dilation 1, stride 1, non-causal) neighbourhood attention: use the direct row-chunked
        # version unless PIXAL_NA_IMPL=natten. Exact, and far faster than flex on gfx1151.
        if (fn.__name__ == "na2d" and len(args) >= 3 and os.environ.get("PIXAL_NA_IMPL", "torch") == "torch"
                and kwargs.get("stride", 1) in (1, (1, 1)) and not kwargs.get("is_causal", False)
                and args[0].dim() == 5 and args[0].shape[1:3] == args[1].shape[1:3] == args[2].shape[1:3]):
            ks = kwargs.get("kernel_size", args[3] if len(args) > 3 else None)
            dil = kwargs.get("dilation", 1)
            dh, dw = (dil, dil) if isinstance(dil, int) else tuple(dil)
            q, k, v = args[:3]
            try:
                if dh == dw == 1:
                    return _local_na2d(q, k, v, ks, scale=kwargs.get("scale"))
                # dilation d: NATTEN runs plain neighbourhood attention inside each interleaved sub-grid.
                # PIXAL_NA_BATCH=1: fold the dh*dw sub-grids into the batch for one call. Measured 2026-11-14 at
                # 1024^2, dilation 16: 15.2 s / 21.4 GiB peak vs 17.8 s / 8.4 GiB for the per-sub-grid loop, so the
                # loop stays the default (the GPU may be shared with other apps).
                B_, H_, W_, nh = q.shape[:4]
                if os.environ.get("PIXAL_NA_BATCH", "0") == "1" and H_ % dh == 0 and W_ % dw == 0:
                    def fold(t):
                        return (t.reshape(B_, H_ // dh, dh, W_ // dw, dw, nh, t.shape[-1]).permute(0, 2, 4, 1, 3, 5, 6)
                                 .reshape(B_ * dh * dw, H_ // dh, W_ // dw, nh, t.shape[-1]))
                    o = _local_na2d(fold(q), fold(k), fold(v), ks, scale=kwargs.get("scale"))
                    return (o.reshape(B_, dh, dw, H_ // dh, W_ // dw, nh, o.shape[-1]).permute(0, 3, 1, 4, 2, 5, 6)
                             .reshape(B_, H_, W_, nh, o.shape[-1]))
                out = torch.empty(*q.shape[:-1], v.shape[-1], device=q.device, dtype=v.dtype)
                for a in range(dh):
                    for b in range(dw):
                        out[:, a::dh, b::dw] = _local_na2d(q[:, a::dh, b::dw], k[:, a::dh, b::dw],
                                                           v[:, a::dh, b::dw], ks, scale=kwargs.get("scale"))
                return out
            except ValueError:
                pass
        if "backend" in kwargs:
            kwargs["backend"] = _flex_backend(kwargs["backend"])
        if str(kwargs.get("backend", "flex")).startswith("flex"):
            # Compiled flex builds the neighbourhood mask block-sparsely; uncompiled flex materialises a dense
            # Q x KV mask (13 GiB at 128x128, 512 GiB at NAF's size). Compiled flex takes fp16/bf16 only, so
            # fp32 inputs run in fp16 and the result is cast back.
            kwargs.setdefault("torch_compile", True)
            if kwargs["torch_compile"] and len(args) >= 3 and args[0].dtype == torch.float32:
                out = wrapper(*(a.half() if torch.is_tensor(a) else a for a in args[:3]), *args[3:], **kwargs)
                return out.float()
        # flex backends need equal head dims for QK and V; NAF uses 64 (QK) / 256 (V). The attention weights
        # depend only on Q and K, so run V in QK-sized slices (zero-padded if needed) and join: exact.
        if len(args) >= 3:
            q, k, v = args[:3]
            dq, dv = q.shape[-1], v.shape[-1]
            if dv != dq:
                pad = (-dv) % dq
                vp = torch.nn.functional.pad(v, (0, pad)) if pad else v
                outs = [fn(q, k, chunk.contiguous(), *args[3:], **kwargs) for chunk in vp.split(dq, dim=-1)]
                return torch.cat(outs, dim=-1)[..., :dv]
        return fn(*args, **kwargs)
    wrapper._pixal3d_rocm = True
    return wrapper


def install_natten() -> bool:
    if not torch.version.hip or os.environ.get("PIXAL_ROCM_NATTEN", "1") == "0":
        return False
    try:
        import natten
        import natten.functional as nf
    except ImportError:
        return False
    if natten.HAS_LIBNATTEN:
        return False
    # natten only reads the compute capability on CUDA builds (torch.version.cuda), so on ROCm it decides
    # torch.compile is unsupported; PyTorch 2.11's compiler (Triton) works on gfx1151. Opt in explicitly.
    import natten.backends.configs.checks as _checks
    _checks._FLEX_COMPILE_SUPPORTED = True
    natten.allow_flex_compile(True)
    # natten passes 64-wide tiles and then compiles flex with mode="max-autotune-no-cudagraphs", which spent 25+
    # minutes benchmarking kernel variants on gfx1151 without finishing (2026-11-14). Keep the tiles (the block
    # mask is built for them) but compile in the default mode: one kernel with BLOCK_M/BLOCK_N = tile size.
    # PIXAL_FLEX_AUTOTUNE=1 restores natten's autotuning.
    import natten.backends.flex as _flex
    if not getattr(_flex.get_flex_attention_fn, "_pixal3d_rocm", False):
        _orig = _flex.get_flex_attention_fn

        @functools.wraps(_orig)
        def _get_flex_attention_fn(torch_compile, torch_compile_args=None):
            if torch_compile_args and os.environ.get("PIXAL_FLEX_AUTOTUNE", "0") != "1":
                torch_compile_args = {k: v for k, v in torch_compile_args.items() if k != "mode"}
            return _orig(torch_compile, torch_compile_args)
        _get_flex_attention_fn._pixal3d_rocm = True
        _flex.get_flex_attention_fn = _get_flex_attention_fn
    for name in ("na1d", "na2d", "na3d"):
        for mod in (natten, nf):
            fn = getattr(mod, name, None)
            if fn is not None and not getattr(fn, "_pixal3d_rocm", False):
                setattr(mod, name, _wrap(fn))
    return True


def cap_gpu_memory() -> None:
    """Cap this process's GPU memory (PIXAL_GPU_MEM_GB, default 40; 0 = no cap). On APUs like Strix Halo the GPU borrows
    system memory (small BIOS carve-out + large GTT) that other GPU apps share, and ROCm reports the whole GTT as
    free even when another process holds much of it. A typical model peaks at ~20 GB; past the cap
    PyTorch raises an out-of-memory error here instead of the kernel's OOM killer picking another process."""
    if not torch.version.hip or not torch.cuda.is_available():
        return
    cap_gb = float(os.environ.get("PIXAL_GPU_MEM_GB", "40"))
    if cap_gb > 0:
        total = torch.cuda.get_device_properties(0).total_memory
        torch.cuda.set_per_process_memory_fraction(min(1.0, cap_gb * 2 ** 30 / total))


def install() -> None:
    from . import torch_rasterizer
    torch_rasterizer.install()
    install_natten()
    cap_gpu_memory()
