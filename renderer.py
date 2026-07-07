"""Render the gasket to a PIL image. Black background, stroke dims with depth."""
from __future__ import annotations

from PIL import Image, ImageDraw


def render(circles_with_depth, size: int, max_depth: int = 8) -> Image.Image:
    canvas = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(canvas)

    for circle, depth in circles_with_depth:
        r = circle.r
        cx, cy = circle.cx, circle.cy
        # deeper => dimmer stroke, with a 15% floor so tiny circles remain visible
        intensity = max(0.15, 1.0 - depth / max(max_depth, 1))
        v = int(255 * intensity)
        color = (v, v, v)

        if r < 1.0:
            draw.point((cx, cy), fill=color)
            continue

        bbox = [cx - r, cy - r, cx + r, cy + r]
        width = 2 if depth == 0 else 1
        draw.ellipse(bbox, outline=color, width=width)

    return canvas
