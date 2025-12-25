#!/usr/bin/env python3
"""
Image to Markdown converter using OCR.

Backends:
1. Surya OCR - Good quality, comes with Marker
2. Tesseract - Traditional OCR, widely available
3. EasyOCR - Deep learning based
"""

import argparse
from pathlib import Path
from typing import Optional


def convert_with_surya(image_path: Path, **kwargs) -> str:
    """Convert image using Surya OCR (installed with Marker)."""
    from surya.recognition import RecognitionPredictor
    from surya.detection import DetectionPredictor
    from PIL import Image

    # Load image
    image = Image.open(image_path)

    # Initialize predictors
    det_predictor = DetectionPredictor()
    rec_predictor = RecognitionPredictor()

    # Detect text regions
    det_results = det_predictor([image])

    # Recognize text
    rec_results = rec_predictor([image], det_results)

    # Extract text
    lines = []
    for result in rec_results:
        for line in result.text_lines:
            lines.append(line.text)

    return "\n".join(lines)


def convert_with_tesseract(image_path: Path, **kwargs) -> str:
    """Convert image using Tesseract OCR."""
    try:
        import pytesseract
        from PIL import Image

        image = Image.open(image_path)
        text = pytesseract.image_to_string(image, lang=kwargs.get('lang', 'eng'))
        return text.strip()
    except ImportError:
        raise ImportError("pytesseract not installed. Run: pip install pytesseract")


def convert_with_easyocr(image_path: Path, **kwargs) -> str:
    """Convert image using EasyOCR."""
    try:
        import easyocr

        reader = easyocr.Reader(['en', 'ch_tra'])  # English and Traditional Chinese
        results = reader.readtext(str(image_path))

        lines = [text for _, text, _ in results]
        return "\n".join(lines)
    except ImportError:
        raise ImportError("easyocr not installed. Run: pip install easyocr")


def convert_image(
    image_path: Path,
    backend: str = "auto",
    **kwargs
) -> str:
    """
    Convert image to text/Markdown using OCR.

    Args:
        image_path: Path to image file
        backend: "auto", "surya", "tesseract", or "easyocr"
        **kwargs: Backend-specific options

    Returns:
        Extracted text
    """
    backends = {
        "surya": convert_with_surya,
        "tesseract": convert_with_tesseract,
        "easyocr": convert_with_easyocr,
    }

    if backend == "auto":
        # Try backends in order
        for name in ["tesseract", "surya", "easyocr"]:
            try:
                return backends[name](image_path, **kwargs)
            except ImportError:
                continue
            except Exception as e:
                print(f"Warning: {name} failed: {e}")
                continue
        raise RuntimeError("No OCR backend available. Install pytesseract or easyocr.")

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}")

    return backends[backend](image_path, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Convert image to text using OCR"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input image file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output text file (default: stdout)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "surya", "tesseract", "easyocr"],
        default="auto",
        help="OCR backend (default: auto)"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="OCR language (for tesseract, default: eng)"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}")
        return 1

    try:
        text = convert_image(
            args.input,
            backend=args.backend,
            lang=args.lang,
        )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text, encoding="utf-8")
            print(args.output)
        else:
            print(text)

        return 0

    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
