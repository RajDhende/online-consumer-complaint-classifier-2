"""Generate the project logo (run once; the PNG is committed)."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

SIZE = 512
SCALE = 4  # supersample for smooth edges
S = SIZE * SCALE


def rounded_mask(size: int, radius: int) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size - 1, size - 1), radius=radius, fill=255
    )
    return mask


def gradient(
    size: int, start: tuple[int, int, int], end: tuple[int, int, int]
) -> Image.Image:
    image = Image.new("RGB", (size, size))
    pixels = image.load()
    for y in range(size):
        ratio = y / max(size - 1, 1)
        color = tuple(int(start[i] + (end[i] - start[i]) * ratio) for i in range(3))
        for x in range(size):
            pixels[x, y] = color
    return image


def main() -> None:
    canvas = Image.new("RGBA", (S, S), (0, 0, 0, 0))

    # Rounded gradient tile with a soft inner highlight.
    tile = gradient(S, (30, 27, 75), (67, 56, 202)).convert("RGBA")
    highlight = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(highlight)
    hd.ellipse((-S * 0.25, -S * 0.55, S * 0.85, S * 0.35), fill=(6, 182, 212, 90))
    tile = Image.alpha_composite(tile, highlight)
    canvas.paste(tile, (0, 0), rounded_mask(S, int(S * 0.22)))

    draw = ImageDraw.Draw(canvas)

    # Speech bubble.
    left, top, right, bottom = (
        int(S * 0.20),
        int(S * 0.24),
        int(S * 0.80),
        int(S * 0.66),
    )
    draw.rounded_rectangle(
        (left, top, right, bottom),
        radius=int(S * 0.085),
        fill=(255, 255, 255, 255),
    )
    draw.polygon(
        [
            (int(S * 0.33), bottom - int(S * 0.02)),
            (int(S * 0.31), int(S * 0.78)),
            (int(S * 0.47), bottom - int(S * 0.02)),
        ],
        fill=(255, 255, 255, 255),
    )

    # Check mark inside the bubble.
    width = int(S * 0.045)
    draw.line(
        [
            (int(S * 0.36), int(S * 0.45)),
            (int(S * 0.46), int(S * 0.55)),
            (int(S * 0.65), int(S * 0.35)),
        ],
        fill=(79, 70, 229, 255),
        width=width,
        joint="curve",
    )

    # Small data bars under the bubble.
    bar_color = (165, 180, 252, 235)
    for index, height in enumerate([0.07, 0.11, 0.16, 0.22]):
        x = int(S * (0.30 + index * 0.11))
        draw.rounded_rectangle(
            (x, int(S * 0.78) - int(S * height), x + int(S * 0.055), int(S * 0.78)),
            radius=int(S * 0.012),
            fill=bar_color,
        )

    logo = canvas.resize((SIZE, SIZE), Image.LANCZOS)
    output = Path(__file__).resolve().parent.parent / "assets" / "logo.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    logo.save(output)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
