from PIL import Image, ImageFilter
import numpy as np
import os
import random

INPUT_DIR = "pre/dots"
OUTPUT_DIR = "dataset/dot"

os.makedirs(OUTPUT_DIR, exist_ok=True)

GENERATED_PER_IMAGE = 20

info = 0

def add_noise(img, amount=8):
    arr = np.array(img).astype(np.int16)

    noise = np.random.normal(0, amount, arr.shape)

    arr = arr + noise
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    return Image.fromarray(arr)

for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img = Image.open(os.path.join(INPUT_DIR, filename)).convert("L")
    arr = np.array(img)

    mask = arr < 220

    if not mask.any():
        continue

    y, x = np.where(mask)

    crop = arr[
        y.min():y.max()+1,
        x.min():x.max()+1
    ]

    crop = Image.fromarray(crop)

    for variation in range(GENERATED_PER_IMAGE):

        canvas = Image.new("L", (90, 140), 255)

        w, h = crop.size

        target_size = random.randint(12, 25)

        scale = min(target_size / w, target_size / h)

        resized = crop.resize(
            (
                max(1, int(w * scale)),
                max(1, int(h * scale))
            ),
            Image.Resampling.LANCZOS
        )

        angle = random.uniform(-30, 30)

        resized = resized.rotate(
            angle,
            expand=True,
            fillcolor=255
        )

        x_pos = random.randint(5, 90 - resized.width - 5)
        y_pos = random.randint(5, 140 - resized.height - 5)

        canvas.paste(resized, (x_pos, y_pos))

        if random.random() < 0.7:
            canvas = add_noise(
                canvas,
                random.uniform(2, 12)
            )

        if random.random() < 0.5:
            canvas = canvas.filter(
                ImageFilter.GaussianBlur(
                    random.uniform(0.2, 1.0)
                )
            )

        output_name = (
            f"{os.path.splitext(filename)[0]}"
            f"_{variation}.jpg"
        )

        canvas.convert("RGB").save(
            os.path.join(OUTPUT_DIR, output_name),
            quality=95
        )

        info += 1

print(f"imagenes totales: {info}")