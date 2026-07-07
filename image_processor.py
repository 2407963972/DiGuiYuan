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
