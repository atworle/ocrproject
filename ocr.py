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
    """Run OCR with simple grayscale + binary thresholding (cleaner for historical prints)."""
    try:
        import pytesseract
        from PIL import Image, ImageEnhance, ImageFilter

        # Load and preprocess
        img = Image.open(image_path).convert('L')  # grayscale
        img = img.filter(ImageFilter.MedianFilter())  # light denoise
        img = ImageEnhance.Contrast(img).enhance(2)  # boost contrast

        # Simple binary threshold — crucial for faint 18th-c. type
        bw = img.point(lambda x: 0 if x < 160 else 255, '1')

        # OCR config: OEM 3 (LSTM), PSM 6 (block of text)
        config = f'--oem 3 --psm 6 -l {language}'
        text = pytesseract.image_to_string(bw, config=config)

        # Optional confidence extraction
        try:
            data = pytesseract.image_to_data(bw, config=config, output_type=pytesseract.Output.DICT)
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
