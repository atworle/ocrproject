#!/usr/bin/env python3
"""
Standalone OCR script for preprocessed images
=============================================

This script runs Tesseract OCR on preprocessed images (grayscale, denoised, etc.)
and saves the text output to a specified directory.

Usage:
    python ocr_pipeline.py --input-dir /path/to/imagesprocessed --output-dir /path/to/ocr-results --batch
"""

import sys
import argparse
from pathlib import Path
import re
import cv2 
from PIL import Image 

def check_dependencies():
    """Check if required packages and Tesseract are installed."""
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        print("❌ Missing required packages: install pytesseract and Pillow")
        print("   pip install pytesseract Pillow")
        return False

    try:
        import subprocess
        result = subprocess.run(['tesseract', '--version'], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise FileNotFoundError
        return True
    except Exception:
        print("❌ Tesseract OCR not found. Install it:")
        print("   macOS: brew install tesseract")
        print("   Ubuntu: sudo apt-get install tesseract-ocr")
        print("   Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        return False

def run_ocr_psm3(image_path, language='eng'):
    """Run OCR with PSM 3 mode, including upscaling, grayscale, and adaptive thresholding."""
    try:
        import pytesseract
        import cv2
        from PIL import Image
        import numpy as np

        # Load image
        img_cv = cv2.imread(str(image_path))

        # Upscale (2-4x depending on size)
        scale_factor = 3
        img_large = cv2.resize(img_cv, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Convert to grayscale
        gray = cv2.cvtColor(img_large, cv2.COLOR_BGR2GRAY)

        # Optional: histogram equalization or CLAHE for contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        kernel = np.array([[0,-0.5,0], [-0.5,3,-0.5], [0,-0.5,0]])

        gray = cv2.filter2D(gray, -1, kernel)

        # Adaptive thresholding to binarize
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 3
        )

        # Optional: denoise (median filter)
        processed = cv2.medianBlur(thresh, 3)

        # Convert to PIL Image
        image = Image.fromarray(processed)

        # Configure Tesseract
        config = f'--oem 3 --psm 6 -l {language}'

        # Run OCR
        text = pytesseract.image_to_string(image, config=config)

        # Get confidence data
        try:
            data = pytesseract.image_to_data(image, config=config, output_type=pytesseract.Output.DICT)
        except Exception:
            data = {'conf': []}

        return text, data

    except Exception as e:
        print(f"❌ Error running OCR: {str(e)}")
        return None, None

def analyze_confidence(data):
    """Analyze OCR confidence scores."""
    if not data or 'conf' not in data:
        return None
    confidences = [int(c) for c in data['conf'] if int(c) > 0]
    if not confidences:
        return None
    return {
        'mean': sum(confidences) / len(confidences),
        'min': min(confidences),
        'max': max(confidences),
        'count': len(confidences)
    }

def clean_text(text):
    """Clean OCR output."""
    if not text:
        return ""
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
    return text

def process_image(image_path, output_dir):
    """Run OCR on a single image and save results."""
    text, data = run_ocr_psm3(image_path)
    if text is None:
        print(f"❌ OCR failed for {image_path.name}")
        return None
    cleaned = clean_text(text)
    confidence = analyze_confidence(data)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{image_path.stem}.txt"
    with open(out_file, 'w', encoding='utf-8') as f:
        f.write(f"Image: {image_path.name}\n")
        f.write(f"Word count: {len(cleaned.split())}\n")
        if confidence:
            f.write(f"Confidence: {confidence['mean']:.1f}% (min: {confidence['min']}, max: {confidence['max']})\n")
        f.write("-" * 50 + "\n\n")
        f.write(cleaned)
    print(f"✅ OCR saved: {out_file.name}")
    return cleaned

def main():
    parser = argparse.ArgumentParser(description="OCR preprocessed images")
    parser.add_argument("--input-dir", required=True, help="Directory of preprocessed images")
    parser.add_argument("--output-dir", required=True, help="Directory to save OCR results")
    parser.add_argument("--batch", action="store_true", help="Process all images in input directory")
    args = parser.parse_args()

    if not check_dependencies():
        return 1

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    imgs = list(input_dir.glob("*.png")) + list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.jpeg"))
    if not imgs:
        print(f"❌ No images found in {input_dir}")
        return 1

    if args.batch:
        print(f"Processing {len(imgs)} images in batch...")
        for img in imgs:
            process_image(img, output_dir)
    else:
        # Interactive mode: select one image
        print("Available images:")
        for i, img in enumerate(imgs, 1):
            print(f"{i}. {img.name}")
        while True:
            choice = input(f"Select image (1-{len(imgs)}) or q to quit: ")
            if choice.lower() in ("q", "quit", "exit"):
                return 0
            try:
                n = int(choice)
                if 1 <= n <= len(imgs):
                    process_image(imgs[n-1], output_dir)
                    break
            except ValueError:
                print("Enter a valid number")

    print("\nOCR processing complete!")
    return 0

if __name__ == "__main__":
    sys.exit(main())
