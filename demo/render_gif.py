"""Render a captured terminal session (with ANSI colors) into an animated GIF.

    bash demo/run_demo.sh > runs/demo.log 2>&1
    uv run python demo/render_gif.py --cmd "bash demo/run_demo.sh" --log runs/demo.log --out docs/demo.gif

The command is "typed", then the log streams line by line in a scrolling window,
with pauses after each section so a reader can follow. Needs Pillow (dev group).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

BG, FG = (24, 26, 33), (220, 223, 228)
COLORS = {
    "31": (240, 98, 98),
    "32": (106, 214, 120),
    "33": (235, 196, 90),
    "90": (128, 133, 145),
    "36": (98, 200, 230),
}
FONTS = [
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/SFNSMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
]
BOLD_FONTS = ["/System/Library/Fonts/Menlo.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"]
SGR = re.compile(r"\x1b\[([0-9;]*)m")


def load_font(paths, size, index=0):
    for p in paths:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size, index=index)
            except OSError:
                continue
    return ImageFont.load_default()


def parse_line(line: str, state=(FG, False, False)):
    """-> (spans, end_state). Colors carry over to the next line, like a real terminal."""
    color, bold, dim = state
    spans, pos = [], 0
    for m in SGR.finditer(line):
        if m.start() > pos:
            spans.append((line[pos : m.start()], color, bold, dim))
        for code in (m.group(1) or "0").split(";"):
            if code in ("0", ""):
                color, bold, dim = FG, False, False
            elif code == "1":
                bold = True
            elif code == "2":
                dim = True
            elif code in COLORS:
                color = COLORS[code]
        pos = m.end()
    if pos < len(line):
        spans.append((line[pos:], color, bold, dim))
    return spans, (color, bold, dim)


class Term:
    def __init__(self, cols, rows, size):
        self.font = load_font(FONTS, size)
        self.bold = load_font(
            BOLD_FONTS,
            size,
            index=1 if BOLD_FONTS[0].endswith(".ttc") and Path(BOLD_FONTS[0]).exists() else 0,
        )
        self.cw = self.font.getbbox("M")[2]
        self.lh = int(size * 1.35)
        self.cols, self.rows = cols, rows
        self.pad, self.bar = 18, 30
        self.w = self.cw * cols + 2 * self.pad
        self.h = self.lh * rows + 2 * self.pad + self.bar

    def frame(self, lines, title):
        img = Image.new("RGB", (self.w, self.h), BG)
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, self.w, self.bar], fill=(40, 43, 52))
        for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([14 + i * 20, 9, 26 + i * 20, 21], fill=c)
        tw = d.textlength(title, font=self.font)
        d.text(((self.w - tw) / 2, 6), title, font=self.font, fill=(160, 165, 175))
        y = self.bar + self.pad
        state = (FG, False, False)
        visible = len(lines) - self.rows
        for i, line in enumerate(lines):
            spans, state_after = parse_line(line, state)
            state = state_after
            if i < visible:
                continue
            x = self.pad
            for text, color, bold, dim in spans:
                text = text[: max(0, self.cols - int((x - self.pad) / self.cw))]
                if not text:
                    continue
                if dim:
                    color = tuple(int(c * 0.6 + b * 0.4) for c, b in zip(color, BG))
                d.text((x, y), text, font=self.bold if bold else self.font, fill=color)
                x += d.textlength(text, font=self.font)
            y += self.lh
        return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cmd", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--out", default="docs/demo.gif")
    p.add_argument("--title", default="mswe-rl-env — offline demo")
    p.add_argument("--cols", type=int, default=120)
    p.add_argument("--rows", type=int, default=30)
    p.add_argument("--font-size", type=int, default=14)
    a = p.parse_args()

    log = Path(a.log).read_text().rstrip("\n").splitlines()
    t = Term(a.cols, a.rows, a.font_size)
    frames, durs = [], []
    prompt = "\x1b[36m~/mswe-rl-env\x1b[0m \x1b[1m$\x1b[0m "

    def add(lines, ms):
        frames.append(t.frame(lines, a.title))
        durs.append(ms)

    add([prompt], 600)
    for i in range(0, len(a.cmd) + 1, 2):
        add([prompt + a.cmd[:i] + "█"], 45)
    shown = [prompt + a.cmd]
    add(shown, 500)
    for line in log:
        shown.append(line)
        plain = SGR.sub("", line)
        if plain.startswith("== "):
            ms = 1200
        elif "throughput" in plain or plain.startswith("episode "):
            ms = 2200
        elif plain.lstrip().startswith(("pass ", "fail ", "env_error")):
            ms = 260
        else:
            ms = 70
        add(shown, ms)
    add(shown, 4000)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # One fixed palette: every text color blended into both backgrounds (for antialiasing).
    # Per-frame adaptive quantization drops small colored areas like "pass"/"fail".
    BAR = (40, 43, 52)
    inks = [FG, (160, 165, 175), (255, 95, 86), (255, 189, 46), (39, 201, 63), *COLORS.values()]
    inks += [tuple(int(c * 0.6 + b * 0.4) for c, b in zip(k, BG)) for k in list(COLORS.values()) + [FG]]
    cols = {BG, BAR}
    for ink in inks:
        for base in (BG, BAR):
            for t in range(1, 9):
                cols.add(tuple(int(b + (k - b) * t / 8) for k, b in zip(ink, base)))
    cols = sorted(cols)[:256]
    pal_img = Image.new("P", (1, 1))
    pal_img.putpalette([v for c in cols for v in c] + [0] * (768 - 3 * len(cols)))
    pal = [f.quantize(palette=pal_img, dither=Image.Dither.NONE) for f in frames]
    pal[0].save(out, save_all=True, append_images=pal[1:], duration=durs, loop=0, optimize=True, disposal=1)
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
