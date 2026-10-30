"""Real non-photo images made by code: text pages, UI screens, charts, logos.

No neural network touches these pixels, so they are real. The "train" split is a small share of the
training reals, against false positives where the product hurts most (screenshots, documents, logos);
the price: without generated counterparts they push towards "graphics = real", so generated logos and
infographics are caught less. The "test" split, with other fonts, is used for calibration and to measure
those false positives. Fonts ship with matplotlib.
"""

import io
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.figure import Figure  # drawn off-screen without pyplot: the notebook backend stays as it is
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
FONTS = {
    "train": ["DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf", "DejaVuSerif-Italic.ttf"],
    "test": ["DejaVuSansMono.ttf", "STIXGeneral.ttf", "STIXGeneralBol.ttf", "cmr10.ttf", "cmss10.ttf"],
}
KINDS = ("text", "ui", "chart", "logo")
LATIN, CYRILLIC = "abcdefghijklmnopqrstuvwxyz", "абвгдеёжзийклмнопрстуфхцчшщыэюя"


def words(rng, n: int) -> str:
    alphabet = CYRILLIC if rng.random() < 0.4 else LATIN
    out = []
    for _ in range(n):
        w = "".join(rng.choice(list(alphabet), size=rng.integers(2, 10)))
        out.append(w.capitalize() if rng.random() < 0.15 else w)
        if rng.random() < 0.05:
            out.append(str(rng.integers(0, 10_000)))
    return " ".join(out)


def font(rng, split: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / rng.choice(FONTS[split])), int(size))


def color(rng, light: bool) -> tuple:
    lo, hi = (200, 256) if light else (0, 120)
    return tuple(int(c) for c in rng.integers(lo, hi, 3))


def text_page(rng, w, h, split):
    img = Image.new("RGB", (w, h), color(rng, light=True))
    d = ImageDraw.Draw(img)
    y, size = int(h * 0.05), int(rng.integers(12, 24))
    d.text((int(w * 0.06), y), words(rng, 4), font=font(rng, split, size * 1.8), fill=color(rng, False))
    y += int(size * 3)
    line_font, ink = font(rng, split, size), color(rng, False)
    while y < h - size * 2:
        if rng.random() < 0.08:  # table rule or empty line
            d.line((int(w * 0.06), y, int(w * 0.94), y), fill=ink, width=1)
        else:
            d.text((int(w * 0.06), y), words(rng, int(w / (size * 3))), font=line_font, fill=ink)
        y += int(size * rng.uniform(1.3, 1.8))
    return img


def ui_screen(rng, w, h, split):
    img = Image.new("RGB", (w, h), color(rng, light=True))
    d = ImageDraw.Draw(img)
    bar, accent = int(h * rng.uniform(0.06, 0.1)), color(rng, light=False)
    d.rectangle((0, 0, w, bar), fill=accent)
    d.text((16, bar // 4), words(rng, 2), font=font(rng, split, bar // 2), fill=(255, 255, 255))
    side = int(w * rng.uniform(0, 0.25))
    if side:
        d.rectangle((0, bar, side, h), fill=color(rng, light=True))
        for i in range(8):
            d.text((12, bar + 20 + i * 36), words(rng, 1), font=font(rng, split, 16), fill=color(rng, False))
    y = bar + 16
    while y < h - 60:  # rows of cards with a title, two lines of text and a button
        x = side + 16
        card_h = int(rng.integers(90, 160))
        while x < w - 140:
            card_w = int(rng.integers(140, 320))
            d.rounded_rectangle((x, y, x + card_w, y + card_h), radius=int(rng.integers(0, 14)),
                                fill=color(rng, light=True), outline=color(rng, light=False))
            d.text((x + 10, y + 8), words(rng, 2), font=font(rng, split, 18), fill=color(rng, False))
            for line in range(2):
                d.text((x + 10, y + 36 + 18 * line), words(rng, 3), font=font(rng, split, 13), fill=color(rng, False))
            d.rounded_rectangle((x + 10, y + card_h - 34, x + 90, y + card_h - 10), radius=6, fill=accent)
            d.text((x + 18, y + card_h - 31), words(rng, 1)[:8], font=font(rng, split, 13), fill=(255, 255, 255))
            x += card_w + 16
        y += card_h + 16
    return img


def chart(rng, w, h, split):
    fig = Figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_subplot()
    x = np.arange(int(rng.integers(5, 30)))
    kind = rng.choice(["line", "bar", "scatter", "pie"])
    if kind == "line":
        for _ in range(int(rng.integers(1, 4))):
            ax.plot(x, np.cumsum(rng.normal(size=len(x))), marker=rng.choice(["", "o", "s"]))
    elif kind == "bar":
        ax.bar(x, rng.uniform(1, 10, len(x)))
    elif kind == "scatter":
        ax.scatter(rng.normal(size=200), rng.normal(size=200), s=8)
    else:
        ax.pie(rng.uniform(1, 5, int(rng.integers(3, 7))))
    ax.set_title(words(rng, 3), family="serif" if split == "test" else "sans-serif")
    ax.grid(rng.random() < 0.5)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    return Image.open(buf).convert("RGB").resize((w, h), Image.NEAREST)


def logo(rng, w, h, split):
    img = Image.new("RGB", (w, h), color(rng, light=rng.random() < 0.7))
    d = ImageDraw.Draw(img)
    for _ in range(int(rng.integers(1, 5))):
        cx, cy, r = rng.integers(w // 4, 3 * w // 4), rng.integers(h // 4, 3 * h // 4), rng.integers(w // 10, w // 3)
        box = (int(cx - r), int(cy - r), int(cx + r), int(cy + r))
        shape = rng.choice(["ellipse", "rect", "poly"])
        if shape == "ellipse":
            d.ellipse(box, fill=color(rng, False))
        elif shape == "rect":
            d.rectangle(box, fill=color(rng, False))
        else:
            angles = np.sort(rng.uniform(0, 2 * np.pi, int(rng.integers(3, 7))))
            d.polygon([(float(cx + r * np.cos(a)), float(cy + r * np.sin(a))) for a in angles], fill=color(rng, False))
    d.text((w // 10, int(h * 0.75)), words(rng, 1).upper(), font=font(rng, split, h // 8), fill=color(rng, False))
    return img


DRAW = {"text": text_page, "ui": ui_screen, "chart": chart, "logo": logo}
CONTENT_TYPE = {"text": "text", "ui": "screenshot", "chart": "graphics", "logo": "graphics"}


def render(kind: str, rng, split: str = "train", size=(512, 1024)) -> Image.Image:
    w, h = (int(v) for v in rng.integers(size[0], size[1], 2))
    img = DRAW[kind](rng, w, h, split)
    if kind == "logo" and rng.random() < 0.5:  # anti-aliased shapes: draw at 2x, downsample
        big = DRAW[kind](np.random.default_rng(rng.integers(1 << 31)), 2 * w, 2 * h, split)
        img = big.resize((w, h), Image.LANCZOS)
    return img


def render_folder(out_dir, n: int, split: str = "train", seed: int = 0) -> list[dict]:
    """Writes n images (PNG, a third re-saved as JPEG like shared screenshots). Returns [{path, content}]."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    items = []
    for i in range(n):
        rng = np.random.default_rng([seed, i, 0 if split == "train" else 1])
        kind = KINDS[i % len(KINDS)]
        path = out_dir / f"{split}_{i:05d}_{kind}"
        if not (path.with_suffix(".png").exists() or path.with_suffix(".jpg").exists()):
            img = render(kind, rng, split)
            if rng.random() < 0.33:
                img.save(path.with_suffix(".jpg"), quality=int(rng.integers(80, 96)))
            else:
                img.save(path.with_suffix(".png"))
        file = path.with_suffix(".png") if path.with_suffix(".png").exists() else path.with_suffix(".jpg")
        items.append({"path": file, "content": CONTENT_TYPE[kind]})
    return items
