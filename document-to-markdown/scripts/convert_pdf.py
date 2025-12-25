#!/usr/bin/env python3
"""
PDF to Markdown converter using multiple backends.

Backends (in order of preference):
1. PyMuPDF4LLM - Fast, good for text-based PDFs
2. Marker - Better for complex layouts (requires model download)
3. MarkItDown - Fallback option
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List, Tuple


def log_warning(message: str):
    """Print warning message to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def parse_pages(pages_str: str) -> List[int]:
    """
    Parse page range string into list of page numbers (0-indexed).

    Supports formats:
    - "1-5" → [0, 1, 2, 3, 4]
    - "1,3,5" → [0, 2, 4]
    - "1-3,7,10-12" → [0, 1, 2, 6, 9, 10, 11]
    """
    pages = []
    for part in pages_str.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            start_idx = int(start) - 1  # Convert to 0-indexed
            end_idx = int(end)  # end is inclusive, so don't subtract
            pages.extend(range(start_idx, end_idx))
        else:
            pages.append(int(part) - 1)  # Convert to 0-indexed
    return sorted(set(pages))


def convert_with_pymupdf4llm(pdf_path: Path, **kwargs) -> str:
    """Convert PDF using PyMuPDF4LLM (fast, no model required)."""
    import pymupdf4llm

    # Parse pages if provided as string
    pages = kwargs.get('pages')
    if isinstance(pages, str):
        pages = parse_pages(pages)

    return pymupdf4llm.to_markdown(
        str(pdf_path),
        pages=pages,
        page_chunks=kwargs.get('page_chunks', False),
        write_images=kwargs.get('write_images', False),
        image_path=kwargs.get('image_path'),
        dpi=kwargs.get('dpi', 150),
    )


def convert_with_marker(pdf_path: Path, **kwargs) -> str:
    """Convert PDF using Marker (better for complex layouts, requires models)."""
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    models = create_model_dict()
    converter = PdfConverter(artifact_dict=models)
    result = converter(str(pdf_path))

    return result.markdown


def convert_with_markitdown(pdf_path: Path, **kwargs) -> str:
    """Convert PDF using MarkItDown (fallback)."""
    from markitdown import MarkItDown

    md = MarkItDown()
    result = md.convert(str(pdf_path))
    return result.text_content


def is_content_sufficient(content: str, threshold: int = 100) -> bool:
    """Check if extracted content is sufficient (not a scanned PDF)."""
    if not content:
        return False
    # Strip whitespace and check length
    stripped = content.strip()
    # Also check if it's mostly whitespace/newlines
    non_whitespace = ''.join(stripped.split())
    return len(non_whitespace) >= threshold


def has_malformed_tables(content: str) -> bool:
    """
    Detect if content has table-like patterns that aren't properly formatted.

    Signs of malformed tables:
    - Multiple x/✓ in a row without | separators
    - Column headers followed by data without proper markdown table format
    """
    import re

    lines = content.split('\n')

    for i, line in enumerate(lines):
        # Check for rows with 3+ checkmarks/x without being in a markdown table
        if re.search(r'[x✓✗].*[x✓✗].*[x✓✗]', line, re.IGNORECASE):
            # If it's already a markdown table row, skip
            if line.strip().startswith('|') and line.strip().endswith('|'):
                continue
            # Found malformed table row
            return True

        # Check for number-based tables (e.g., "Model 95.2 87.3 91.5")
        # Pattern: text followed by 3+ numbers separated by spaces
        if re.search(r'\w+\s+\d+\.?\d*\s+\d+\.?\d*\s+\d+\.?\d*', line):
            if not line.strip().startswith('|'):
                return True

    return False


def convert_pdf(
    pdf_path: Path,
    backend: str = "auto",
    **kwargs
) -> Tuple[str, str]:
    """
    Convert PDF to Markdown.

    Args:
        pdf_path: Path to PDF file
        backend: "auto", "pymupdf4llm", "marker", or "markitdown"
        **kwargs: Backend-specific options

    Returns:
        Tuple of (markdown_content, backend_used)
    """
    backends = {
        "pymupdf4llm": convert_with_pymupdf4llm,
        "marker": convert_with_marker,
        "markitdown": convert_with_markitdown,
    }

    if backend == "auto":
        # Try PyMuPDF4LLM first
        try:
            result = convert_with_pymupdf4llm(pdf_path, **kwargs)

            # Check if content is sufficient (not a scanned PDF)
            if not is_content_sufficient(result):
                # Content too short, likely scanned PDF
                log_warning("Content too short, PDF may be scanned. Trying Marker backend...")
                try:
                    result = convert_with_marker(pdf_path, **kwargs)
                    return result, "marker"
                except ImportError:
                    log_warning("Marker not installed. Install with: pip install marker-pdf")
                    return result, "pymupdf4llm"
                except Exception as e:
                    log_warning(f"Marker failed: {e}")
                    return result, "pymupdf4llm"

            # Check for malformed tables and warn (don't auto-retry, Marker is slow)
            if has_malformed_tables(result):
                log_warning("Complex tables detected. For better results, retry with: --pdf-backend marker")

            return result, "pymupdf4llm"

        except ImportError:
            pass
        except Exception as e:
            log_warning(f"PyMuPDF4LLM failed: {e}")

        # Try remaining backends
        for name in ["marker", "markitdown"]:
            try:
                return backends[name](pdf_path, **kwargs), name
            except ImportError:
                continue
            except Exception as e:
                log_warning(f"{name} failed: {e}")
                continue

        raise RuntimeError("No PDF backend available")

    if backend not in backends:
        raise ValueError(f"Unknown backend: {backend}")

    return backends[backend](pdf_path, **kwargs), backend


# Keep backward compatibility - return just string if called directly
def convert_pdf_simple(pdf_path: Path, backend: str = "auto", **kwargs) -> str:
    """Backward-compatible wrapper that returns just the markdown string."""
    result, _ = convert_pdf(pdf_path, backend, **kwargs)
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert PDF to Markdown"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        required=True,
        help="Input PDF file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output Markdown file (default: stdout)"
    )
    parser.add_argument(
        "--backend", "-b",
        choices=["auto", "pymupdf4llm", "marker", "markitdown"],
        default="auto",
        help="Conversion backend (default: auto)"
    )
    parser.add_argument(
        "--pages", "-p",
        type=str,
        help="Page range to convert (e.g., '1-5', '1,3,5', '1-3,7,10-12')"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for image extraction (default: 150)"
    )
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Extract and save images"
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        help="Directory for extracted images"
    )

    args = parser.parse_args()

    if not args.input.exists():
        print(f"Error: File not found: {args.input}", file=sys.stderr)
        return 1

    try:
        markdown, backend_used = convert_pdf(
            args.input,
            backend=args.backend,
            pages=args.pages,
            dpi=args.dpi,
            write_images=args.write_images,
            image_path=str(args.image_path) if args.image_path else None,
        )

        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(markdown, encoding="utf-8")
            print(f"{args.output} (backend: {backend_used})")
        else:
            print(markdown)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
