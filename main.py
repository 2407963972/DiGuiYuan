"""Tkinter UI: left = uploaded image, right = brightness-driven Apollonian gasket."""
from __future__ import annotations

import random
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageTk

from apollonian import build_gasket, make_initial
from image_processor import center_brightness, load_and_process, sample_brightness_region
from renderer import render

CANVAS_SIZE = 1000
BIG_RADIUS = 300
MAX_DEPTH = 10
MIN_RADIUS = 0.5
CENTER_BRIGHT_THRESHOLD = 0.5
DISPLAY_SIZE = 480


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("Brightness-Driven Apollonian Gasket")
        root.geometry(f"{DISPLAY_SIZE * 2 + 60}x{DISPLAY_SIZE + 100}")

        top = tk.Frame(root)
        top.pack(pady=8)
        tk.Button(top, text="选择图片", command=self.on_choose, width=14).pack(side=tk.LEFT, padx=6)
        tk.Button(top, text="重新生成", command=self.on_regenerate, width=14).pack(side=tk.LEFT, padx=6)
        self.status = tk.Label(top, text="请选择一张图片")
        self.status.pack(side=tk.LEFT, padx=12)

        body = tk.Frame(root)
        body.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        left_box = tk.Frame(body)
        left_box.pack(side=tk.LEFT, expand=True)
        tk.Label(left_box, text="原图").pack()
        self.left_label = tk.Label(left_box, bg="#222", width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        self.left_label.pack()

        right_box = tk.Frame(body)
        right_box.pack(side=tk.RIGHT, expand=True)
        tk.Label(right_box, text="生成图").pack()
        self.right_label = tk.Label(right_box, bg="#222", width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        self.right_label.pack()

        self._original: Image.Image | None = None
        self._matrix = None
        self._left_photo = None
        self._right_photo = None

    def on_choose(self) -> None:
        path = filedialog.askopenfilename(
            title="选择一张图片",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.webp"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            original, matrix = load_and_process(path)
        except Exception as exc:
            messagebox.showerror("读取失败", str(exc))
            return

        self._original = original
        self._matrix = matrix
        self._show_original(original)
        self._generate_and_show(seed=None)

    def on_regenerate(self) -> None:
        if self._original is None or self._matrix is None:
            self.status.config(text="请先选择图片")
            return
        self._generate_and_show(seed=None)

    def _show_original(self, img: Image.Image) -> None:
        disp = img.copy()
        disp.thumbnail((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
        self._left_photo = ImageTk.PhotoImage(disp)
        self.left_label.config(image=self._left_photo, width=disp.width, height=disp.height)

    def _generate_and_show(self, seed: int | None) -> None:
        rng = random.Random(seed)
        matrix = self._matrix

        def brightness_fn(cx: float, cy: float, r: float) -> float:
            return sample_brightness_region(matrix, cx, cy, r, CANVAS_SIZE)

        center_val = center_brightness(matrix)
        n = 3 if center_val > CENTER_BRIGHT_THRESHOLD else 2
        self.status.config(text=f"中心亮度 {center_val:.2f} → 初始 {n} 个内切圆，生成中...")
        self.root.update_idletasks()

        big, smalls = make_initial(CANVAS_SIZE, BIG_RADIUS, n, rng)
        circles = build_gasket(
            big, smalls, brightness_fn, CANVAS_SIZE,
            max_depth=MAX_DEPTH, min_r=MIN_RADIUS,
        )
        canvas = render(circles, size=CANVAS_SIZE, max_depth=MAX_DEPTH)
        disp = canvas.resize((DISPLAY_SIZE, DISPLAY_SIZE), Image.LANCZOS)
        self._right_photo = ImageTk.PhotoImage(disp)
        self.right_label.config(image=self._right_photo, width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        self.status.config(text=f"完成：{len(circles)} 个圆，中心亮度 {center_val:.2f}，初始 {n} 圆")


def main() -> int:
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
