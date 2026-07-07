"""Brightness field: load an image, downsample to 4x4, blur, sample bilinearly."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageFilter

GRID = 4


def load_and_process(path: str) -> tuple[Image.Image, np.ndarray]:
    """Return (original PIL image in RGB, blurred 4x4 brightness matrix in [0,1])."""
    original = Image.open(path).convert("RGB")
    gray = original.convert("L")
    small = gray.resize((GRID, GRID), Image.LANCZOS)
    blurred = small.filter(ImageFilter.GaussianBlur(radius=0.8))
    mat = np.asarray(blurred, dtype=np.float64) / 255.0
    return original, mat


def sample_brightness(mat: np.ndarray, x: float, y: float, canvas_size: int) -> float:
    """Bilinear sample of a GRID x GRID matrix at canvas coordinate (x, y).

    Maps canvas [0, canvas_size] to grid cell centers so that the corners of
    the canvas map to the corners of the outermost cells (clamped).
    """
    # Normalize to [0, GRID-1] in matrix-index space
    u = np.clip(x / canvas_size * (GRID - 1), 0.0, GRID - 1)
    v = np.clip(y / canvas_size * (GRID - 1), 0.0, GRID - 1)

    i0 = int(np.floor(v))
    j0 = int(np.floor(u))
    i1 = min(i0 + 1, GRID - 1)
    j1 = min(j0 + 1, GRID - 1)

    dy = v - i0
    dx = u - j0

    a = mat[i0, j0] * (1 - dx) + mat[i0, j1] * dx
    b = mat[i1, j0] * (1 - dx) + mat[i1, j1] * dx
    return float(a * (1 - dy) + b * dy)


def center_brightness(mat: np.ndarray) -> float:
    """Average of the central 2x2 block, used to pick 2 vs 3 initial circles."""
    return float(mat[1:3, 1:3].mean())


def sample_brightness_region(mat: np.ndarray, cx: float, cy: float, r: float,
                              canvas_size: int) -> float:
    """Max brightness inside the circle's bounding box, mapped to matrix indices.

    Small circles (r << cell size) effectively degenerate to point sampling via
    the bilinear fallback; larger circles get the maximum over any matrix cell
    they touch, so a circle whose center sits in a dark cell but whose area
    reaches into a bright cell can still see the bright value and keep recursing.
    """
    n = mat.shape[0]
    scale = (n - 1) / canvas_size
    j_min = int(np.floor(max(0.0, (cx - r) * scale)))
    j_max = int(np.ceil(min(n - 1, (cx + r) * scale)))
    i_min = int(np.floor(max(0.0, (cy - r) * scale)))
    i_max = int(np.ceil(min(n - 1, (cy + r) * scale)))
    j_min = max(0, min(n - 1, j_min))
    j_max = max(0, min(n - 1, j_max))
    i_min = max(0, min(n - 1, i_min))
    i_max = max(0, min(n - 1, i_max))
    region_max = float(mat[i_min:i_max + 1, j_min:j_max + 1].max())
    # blend with the exact bilinear sample so very small circles are point-like
    point_val = sample_brightness(mat, cx, cy, canvas_size)
    # weight by radius vs. one cell width — small r → point-dominated, big r → region-dominated
    cell_size = canvas_size / n
    w = min(1.0, r / cell_size)
    return (1 - w) * point_val + w * region_max
