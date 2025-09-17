import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# -------------------
# Image I/O
# -------------------

def load_image(path):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Could not open image: {path}")
    return img


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# -------------------
# Contrast Enhancement
# -------------------

def apply_hist_eq(img):
    return cv2.equalizeHist(img)


def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)


def apply_gamma(img, gamma=1.2):
    inv_gamma = 1.0 / gamma
    table = np.array([(i / 255.0) ** inv_gamma * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(img, table)


def apply_unsharp(img, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    blurred = cv2.GaussianBlur(img, kernel_size, sigma)
    sharpened = float(amount + 1) * img - float(amount) * blurred
    sharpened = np.clip(sharpened, 0, 255).round().astype(np.uint8)
    if threshold > 0:
        mask = np.absolute(img - blurred) < threshold
        np.copyto(sharpened, img, where=mask)
    return sharpened


# -------------------
# Noise Reduction
# -------------------

def apply_gaussian(img, k=5):
    return cv2.GaussianBlur(img, (k, k), 0)


def apply_median(img, k=5):
    return cv2.medianBlur(img, k)


def apply_bilateral(img, d=9, sc=75, ss=75):
    return cv2.bilateralFilter(img, d, sc, ss)


# -------------------
# Visualization Helpers
# -------------------

def create_comparison_plot(original, processed, titles, save_path=None, show=False):
    n = len(processed) + 1
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).reshape(-1)

    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original")
    axes[0].axis("off")

    for i, (img, t) in enumerate(zip(processed, titles), 1):
        axes[i].imshow(img, cmap="gray")
        axes[i].set_title(t)
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


def create_histogram_comparison(images, titles, save_path=None, show=False):
    fig, axes = plt.subplots(2, len(images), figsize=(5 * len(images), 6))
    for i, (img, t) in enumerate(zip(images, titles)):
        axes[0, i].imshow(img, cmap="gray")
        axes[0, i].set_title(t)
        axes[0, i].axis("off")

        axes[1, i].hist(img.ravel(), bins=256, range=[0, 256], color="blue")
        axes[1, i].set_title(f"Histogram - {t}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -------------------
# Pipelines
# -------------------

def demonstrate_contrast(image_path, output_dir, show=False):
    p = Path(image_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = load_image(p)
    gray = to_gray(img)

    hist_eq = apply_hist_eq(gray)
    clahe = apply_clahe(gray)
    gamma = apply_gamma(gray)
    unsharp = apply_unsharp(gray)

    processed = [hist_eq, clahe, gamma, unsharp]
    titles = ["Hist Eq", "CLAHE", "Gamma", "Unsharp"]

    for im, t in zip(processed, titles):
        cv2.imwrite(str(out / f"{p.stem}_{t.lower().replace(' ', '_')}.png"), im)

    create_comparison_plot(gray, processed, titles,
                           save_path=str(out / f"{p.stem}_contrast_compare.png"),
                           show=show)

    create_histogram_comparison([gray] + processed, ["Original"] + titles,
                                save_path=str(out / f"{p.stem}_histograms.png"),
                                show=show)

    return True


def demonstrate_noise(image_path, output_dir, show=False):
    p = Path(image_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    img = load_image(p)
    gray = to_gray(img)

    g = apply_gaussian(gray)
    m = apply_median(gray)
    b = apply_bilateral(gray)

    processed = [g, m, b]
    titles = ["Gaussian", "Median", "Bilateral"]

    for im, t in zip(processed, titles):
        cv2.imwrite(str(out / f"{p.stem}_{t.lower()}.png"), im)

    create_comparison_plot(gray, processed, titles,
                           save_path=str(out / f"{p.stem}_noise_compare.png"),
                           show=show)

    return True


# -------------------
# CLI
# -------------------

def main():
    parser = argparse.ArgumentParser(description="OCR preprocessing: contrast + noise reduction")
    parser.add_argument("--input-dir", default=str(Path(__file__).parent / "images"))
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "images" / "processed"))
    parser.add_argument("--mode", choices=["contrast", "noise", "all"], default="all",
                        help="Which preprocessing to apply")
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    imgs = [p for p in input_dir.iterdir() if p.is_file()]
    if not imgs:
        print(f"No images found in {input_dir}")
        return 1

    def process(img):
        if args.mode in ("contrast", "all"):
            demonstrate_contrast(img, out_dir / "contrast", show=args.show)
        if args.mode in ("noise", "all"):
            demonstrate_noise(img, out_dir / "noise", show=args.show)

    if args.batch:
        for img in imgs:
            try:
                print(f"Processing {img.name}")
                process(img)
            except Exception as e:
                print(f"Error {img.name}: {e}")
    else:
        print("Found images:")
        for i, img in enumerate(imgs, 1):
            print(f"  {i}. {img.name}")
        while True:
            s = input(f"Select image (1-{len(imgs)} or q): ")
            if s.lower() in ("q", "quit", "exit"):
                return 0
            try:
                n = int(s)
                if 1 <= n <= len(imgs):
                    process(imgs[n - 1])
                    break
            except ValueError:
                print("Enter a number")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
