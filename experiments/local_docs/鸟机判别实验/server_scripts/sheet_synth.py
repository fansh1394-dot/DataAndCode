# -*- coding: utf-8 -*-
"""生成合成帧预览拼版（看气球/风筝贴得是否自然）。"""
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path("/root/rivermind-fs/birddrone/bd_nc3")
files = sorted((OUT / "images/train").glob("*__syn*.jpg")) + \
        sorted((OUT / "images/val").glob("*__syn*.jpg"))
files = files[:48]
cols, cell = 6, 320
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * cell, rows * (cell * 4 // 5 + 14)), "white")
dr = ImageDraw.Draw(sheet)
for i, p in enumerate(files):
    im = Image.open(p).convert("RGB").resize((cell, cell * 4 // 5))
    x, y = (i % cols) * cell, (i // cols) * (cell * 4 // 5 + 14)
    sheet.paste(im, (x, y))
    dr.text((x + 3, y + cell * 4 // 5), p.name[-28:], fill="black")
dst = OUT / "synth_preview.jpg"
sheet.save(dst, quality=90)
print(dst, sheet.size, len(files))
