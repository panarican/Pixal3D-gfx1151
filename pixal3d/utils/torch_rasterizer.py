"""
Pure-PyTorch replacement for nvdiffrast.torch.rasterize, for AMD ROCm.

Why: the HIP build of nvdiffrast's CUDA rasterizer (cudaraster) faults in its coarse-raster stage on
gfx1151 once images grow past ~200 px, and Pixal3D bakes textures in UV space at 1024-2048 px.
Everything else nvdiffrast does (interpolate, texture, antialias) works on ROCm and is kept.

Output matches nvdiffrast's rasterize() (checked against it at small sizes where the HIP build works):
  rast[..., 0:2]  perspective-correct barycentrics (u, v): u weights vertex 0, v vertex 1, 1-u-v vertex 2
  rast[..., 2]    z/w, interpolated linearly in screen space
  rast[..., 3]    triangle index + 1 (0 = empty pixel)
Image row 0 is clip-space y = -1; pixel (i, j) samples clip ((j + .5) / W * 2 - 1, (i + .5) / H * 2 - 1).
The nearest depth wins; equal depths go to the lower triangle index (pixels exactly on a shared edge may
pick the other neighbour than nvdiffrast would, which never changes an interpolated value by more than the
two triangles differ there). No near-plane clipping: triangles with any w <= 0 are skipped. rast_db is
returned as zeros (Pixal3D does not use image-space derivatives).

install() swaps it in for nvdiffrast.torch.rasterize when PyTorch is a ROCm build (PIXAL_TORCH_RASTER=0
turns that off), so o_voxel's texture baking picks it up too without editing that package.
"""
from __future__ import annotations

import os
import torch

_CANDIDATE_BUDGET = int(os.environ.get("PIXAL_RASTER_BUDGET", 24_000_000))  # (triangle, pixel) pairs per chunk
_EMPTY = torch.iinfo(torch.int64).max


def _edge(ax, ay, bx, by, px, py):
    return (bx - ax) * (py - ay) - (by - ay) * (px - ax)


@torch.no_grad()
def _rasterize_one(pos: torch.Tensor, tri: torch.Tensor, H: int, W: int) -> torch.Tensor:
    dev = pos.device
    pos = pos.float()
    tri = tri.long()
    F = tri.shape[0]
    rast = torch.zeros(H, W, 4, device=dev, dtype=torch.float32)
    if F == 0:
        return rast
    v = pos[tri]                                   # (F, 3, 4) clip coords
    w = v[..., 3]
    ok = (w > 0).all(-1)
    ndc = v[..., :3] / torch.where(w > 0, w, torch.ones_like(w)).unsqueeze(-1)
    sx = (ndc[..., 0] * 0.5 + 0.5) * W - 0.5         # pixel-centre coordinates: centre of column j is j
    sy = (ndc[..., 1] * 0.5 + 0.5) * H - 0.5
    area = _edge(sx[:, 0], sy[:, 0], sx[:, 1], sy[:, 1], sx[:, 2], sy[:, 2])
    ok &= area.abs() > 1e-12
    x0 = sx.amin(-1).ceil().clamp(0, W - 1).long(); x1 = sx.amax(-1).floor().clamp(-1, W - 1).long()
    y0 = sy.amin(-1).ceil().clamp(0, H - 1).long(); y1 = sy.amax(-1).floor().clamp(-1, H - 1).long()
    bw = (x1 - x0 + 1).clamp(min=0); bh = (y1 - y0 + 1).clamp(min=0)
    ok &= (bw > 0) & (bh > 0) & (sx.amax(-1) >= -0.5) & (sy.amax(-1) >= -0.5)
    count = torch.where(ok, bw * bh, torch.zeros_like(bw))

    ids = torch.nonzero(count > 0).squeeze(1)
    if ids.numel() == 0:
        return rast
    best = torch.full((H * W,), _EMPTY, device=dev, dtype=torch.int64)
    csum = torch.cumsum(count[ids], 0)
    start = 0
    while start < ids.numel():
        base = csum[start - 1] if start > 0 else torch.zeros((), device=dev, dtype=csum.dtype)
        # take triangles until the candidate budget is used (always at least one)
        end = int(torch.searchsorted(csum, base + _CANDIDATE_BUDGET, right=True).item())
        end = max(end, start + 1)
        t = ids[start:end]
        n = count[t]
        tt = torch.repeat_interleave(t, n)
        first = torch.repeat_interleave(torch.cumsum(n, 0) - n, n)
        local = torch.arange(tt.numel(), device=dev) - first
        px = x0[tt] + local % bw[tt]
        py = y0[tt] + local // bw[tt]
        fx = px.float(); fy = py.float()
        a = area[tt]
        b0 = _edge(sx[tt, 1], sy[tt, 1], sx[tt, 2], sy[tt, 2], fx, fy) / a
        b1 = _edge(sx[tt, 2], sy[tt, 2], sx[tt, 0], sy[tt, 0], fx, fy) / a
        b2 = 1.0 - b0 - b1
        z = b0 * ndc[tt, 0, 2] + b1 * ndc[tt, 1, 2] + b2 * ndc[tt, 2, 2]
        inside = (b0 >= 0) & (b1 >= 0) & (b2 >= 0) & (z >= -1) & (z <= 1)
        if inside.any():
            zq = ((z[inside] + 1) * 0.5 * (2 ** 30)).long().clamp(0, 2 ** 31 - 1)   # depth, monotonic
            key = (zq << 32) | tt[inside]                                       # nearest, then lowest id
            best.scatter_reduce_(0, (py[inside] * W + px[inside]), key, reduce="amin")
        start = end

    hit = torch.nonzero(best != _EMPTY).squeeze(1)
    if hit.numel() == 0:
        return rast
    t = best[hit] & 0xFFFFFFFF
    fy = (hit // W).float(); fx = (hit % W).float()
    a = area[t]
    b0 = _edge(sx[t, 1], sy[t, 1], sx[t, 2], sy[t, 2], fx, fy) / a
    b1 = _edge(sx[t, 2], sy[t, 2], sx[t, 0], sy[t, 0], fx, fy) / a
    b2 = 1.0 - b0 - b1
    z = b0 * ndc[t, 0, 2] + b1 * ndc[t, 1, 2] + b2 * ndc[t, 2, 2]
    q0, q1, q2 = b0 / w[t, 0], b1 / w[t, 1], b2 / w[t, 2]                        # perspective correction
    s = q0 + q1 + q2
    out = torch.stack([q0 / s, q1 / s, z, (t + 1).float()], -1)
    rast.view(H * W, 4)[hit] = out
    return rast


@torch.no_grad()
def rasterize(glctx, pos, tri, resolution, ranges=None, grad_db=True):
    """Same call and return shape as nvdiffrast.torch.rasterize (range mode is not supported)."""
    if ranges is not None:
        raise NotImplementedError("torch_rasterizer: range (instanced) mode is not supported")
    H, W = int(resolution[0]), int(resolution[1])
    if pos.dim() == 2:
        pos = pos.unsqueeze(0)
    rast = torch.stack([_rasterize_one(pos[b], tri, H, W) for b in range(pos.shape[0])])
    return rast, torch.zeros_like(rast)


def install() -> bool:
    """On ROCm builds of PyTorch, route nvdiffrast.torch.rasterize through this module."""
    if not torch.version.hip or os.environ.get("PIXAL_TORCH_RASTER", "1") == "0":
        return False
    try:
        import nvdiffrast.torch as dr
    except ImportError:
        return False
    if getattr(dr.rasterize, "__module__", "") != __name__:
        dr._nvdiffrast_rasterize = dr.rasterize   # kept for comparison tests
        dr.rasterize = rasterize
    return True
