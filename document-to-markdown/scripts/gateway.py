#!/usr/bin/env python3
"""
Gateway - Main entry point for document to Markdown conversion.

Supports:
- PDF: PyMuPDF4LLM (fast), Marker (complex layouts), MarkItDown
- Office: MarkItDown (docx, pptx, xlsx)
- Images: Tesseract, Surya, EasyOCR
- HTML/Web: MarkItDown (local files), Jina Reader (URLs)
- URLs: Jina Reader API (handles JavaScript/SPA)
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Dict, Any, List
from urllib.parse import urlparse

from markitdown import MarkItDown


# File type mappings
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".tiff", ".bmp", ".gif"}
OFFICE_EXTENSIONS = {".docx", ".pptx", ".xlsx", ".doc", ".ppt", ".xls"}
WEB_EXTENSIONS = {".html", ".htm"}
TEXT_EXTENSIONS = {".txt", ".md", ".rst", ".csv", ".json", ".xml"}

# All supported extensions for batch discovery
ALL_SUPPORTED_EXTENSIONS = (
    PDF_EXTENSIONS | IMAGE_EXTENSIONS | OFFICE_EXTENSIONS |
    WEB_EXTENSIONS | TEXT_EXTENSIONS
)


def log_error(message: str):
    """Print error message to stderr."""
    print(f"Error: {message}", file=sys.stderr)


def log_warning(message: str):
    """Print warning message to stderr."""
    print(f"Warning: {message}", file=sys.stderr)


def is_url(path: str) -> bool:
    """Check if path is a URL."""
    try:
        result = urlparse(path)
        return result.scheme in ("http", "https")
    except Exception:
        return False


def detect_file_type(file_path: Path) -> str:
    """Detect file type from extension."""
    suffix = file_path.suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return "pdf"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in OFFICE_EXTENSIONS:
        return "office"
    if suffix in WEB_EXTENSIONS:
        return "web"
    if suffix in TEXT_EXTENSIONS:
        return "text"
    return "unknown"


def extract_pdf_metadata(pdf_path: Path) -> Dict[str, Any]:
    """Extract metadata from PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        metadata = doc.metadata or {}
        page_count = len(doc)
        doc.close()

        return {
            "title": metadata.get("title", "").strip() or None,
            "author": metadata.get("author", "").strip() or None,
            "subject": metadata.get("subject", "").strip() or None,
            "creator": metadata.get("creator", "").strip() or None,
            "producer": metadata.get("producer", "").strip() or None,
            "creation_date": metadata.get("creationDate", "").strip() or None,
            "modification_date": metadata.get("modDate", "").strip() or None,
            "page_count": page_count,
        }
    except Exception:
        return {}


def generate_frontmatter(metadata: Dict[str, Any], source: str) -> str:
    """Generate YAML frontmatter from metadata."""
    lines = ["---"]

    if metadata.get("title"):
        # Escape quotes in title
        title = metadata["title"].replace('"', '\\"')
        lines.append(f'title: "{title}"')

    if metadata.get("author"):
        author = metadata["author"].replace('"', '\\"')
        lines.append(f'author: "{author}"')

    if metadata.get("subject"):
        subject = metadata["subject"].replace('"', '\\"')
        lines.append(f'subject: "{subject}"')

    if metadata.get("page_count"):
        lines.append(f'pages: {metadata["page_count"]}')

    # Add source and conversion timestamp
    lines.append(f'source: "{source}"')
    lines.append(f'converted: "{datetime.now().isoformat()}"')

    lines.append("---")
    lines.append("")  # Blank line after frontmatter

    return "\n".join(lines)


def detect_and_fix_tables(text: str) -> str:
    """
    Detect tables with checkmark/x patterns and format as Markdown tables.
    Handles common academic paper table formats.
    """
    lines = text.split('\n')
    result = []
    i = 0

    while i < len(lines):
        line = lines[i]
        # Check if line has 3+ x/✓ symbols (table row pattern)
        symbols = re.findall(r'[x✓✗]', line, re.IGNORECASE)

        if len(symbols) >= 3:
            # Found table row - look back for header
            header_line = None
            for j in range(len(result) - 1, max(-1, len(result) - 4), -1):
                if j >= 0 and result[j].strip() and not re.search(r'[x✓✗]', result[j], re.IGNORECASE):
                    header_line = result[j].strip()
                    result = result[:j]
                    break

            # Collect all table rows
            table_rows = []
            while i < len(lines):
                row_symbols = re.findall(r'[x✓✗]', lines[i], re.IGNORECASE)
                if len(row_symbols) >= 3:
                    table_rows.append(lines[i].strip())
                    i += 1
                elif lines[i].strip() == '':
                    i += 1
                else:
                    break

            # Format table if we have header and rows
            if header_line and table_rows:
                md_table = _format_checkmark_table(header_line, table_rows)
                result.append(md_table)
            else:
                result.extend(table_rows)
            continue

        result.append(line)
        i += 1

    return '\n'.join(result)


def _format_checkmark_table(header: str, rows: list) -> str:
    """Format detected checkmark table as Markdown."""
    # Determine number of data columns from first row
    first_row = rows[0]
    match = re.match(r'^(.+?)\s+([x✓✗])\s+([x✓✗])\s+([x✓✗])\s*$', first_row, re.IGNORECASE)
    if not match:
        return header + '\n' + '\n'.join(rows)

    num_data_cols = 3

    # Split header intelligently
    header_cols = _smart_split_header(header, num_data_cols)
    header_cols = [''] + header_cols  # Empty first column for model names

    # Build markdown table
    md = '\n| ' + ' | '.join(header_cols) + ' |\n'
    md += '|' + '---|' * len(header_cols) + '\n'

    for row in rows:
        match = re.match(r'^(.+?)\s+([x✓✗])\s+([x✓✗])\s+([x✓✗])\s*$', row, re.IGNORECASE)
        if match:
            name = match.group(1).strip()
            vals = [match.group(i) for i in range(2, 2 + num_data_cols)]
            md += f'| {name} | ' + ' | '.join(vals) + ' |\n'

    return md


def _smart_split_header(header: str, num_cols: int) -> list:
    """Split header text into specified number of columns."""
    # Try double space split first
    parts = re.split(r'\s{2,}', header.strip())
    if len(parts) == num_cols:
        return parts

    # Distribute words evenly
    words = header.split()
    if len(words) >= num_cols:
        cols = []
        words_per_col = len(words) // num_cols
        for i in range(num_cols):
            start = i * words_per_col
            end = start + words_per_col if i < num_cols - 1 else len(words)
            cols.append(' '.join(words[start:end]))
        return cols

    return [header]


def simplify_for_human(markdown: str) -> str:
    """
    Simplify markdown for human reading.
    Removes excessive formatting that's useful for LLMs but noisy for humans.

    Transformations:
    - Remove redundant bold in headers: ## **Title** → ## Title
    - Simplify author lines: **Name** _email_ → Name (email)
    - Clean up redundant link text: [text: url](url) → [text](url)
    - Remove excessive italic markers on common terms
    - Reduce multiple consecutive emphasis markers
    """
    # Remove bold from headers: ## **Title** → ## Title
    # Handle multiple bold sections in same header
    def clean_header(match):
        level = match.group(1)
        content = match.group(2)
        # Remove all ** markers
        content = content.replace('**', '')
        return f"{level} {content.strip()}"

    markdown = re.sub(r'^(#{1,6})\s*(.+)$',
                      lambda m: clean_header(m) if '**' in m.group(2) else m.group(0),
                      markdown, flags=re.MULTILINE)

    # Simplify author format: **Name** _email@domain_ → Name (email@domain)
    markdown = re.sub(
        r'\*\*([^*]+)\*\*\s*_([^_]+@[^_]+)_',
        r'\1 (\2)',
        markdown
    )

    # Clean redundant link text patterns: [text: https://...](https://...) → [text](url)
    markdown = re.sub(
        r'\[([^\]]+?):\s*https?://[^\]]+\]\((https?://[^)]+)\)',
        r'[\1](\2)',
        markdown
    )

    # Remove standalone italic markers around single words that are common terms
    # e.g., _generalist_ → generalist (but keep _emphasized phrase_ intact)
    markdown = re.sub(r'(?<!\w)_([a-zA-Z]+)_(?!\w)', r'\1', markdown)

    # Remove double bold-italic: ***text*** → **text**
    markdown = re.sub(r'\*\*\*([^*]+)\*\*\*', r'**\1**', markdown)

    # Reduce excessive blank lines more aggressively for human reading
    markdown = re.sub(r'\n{3,}', '\n\n', markdown)

    return markdown


def cleanup_markdown(markdown: str) -> str:
    """
    Clean up markdown output:
    - Remove excessive blank lines (3+ → 2)
    - Fix common formatting issues
    - Normalize line endings
    - Fix simple table formatting
    """
    # Normalize line endings
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")

    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in markdown.split("\n")]
    markdown = "\n".join(lines)

    # Replace 3+ consecutive blank lines with 2
    markdown = re.sub(r'\n{4,}', '\n\n\n', markdown)

    # Ensure single blank line before headers
    markdown = re.sub(r'\n{3,}(#{1,6}\s)', r'\n\n\1', markdown)

    # Remove blank lines at start of document
    markdown = markdown.lstrip('\n')

    # Ensure single newline at end of document
    markdown = markdown.rstrip('\n') + '\n'

    # Try to fix simple checkmark/x tables
    markdown = detect_and_fix_tables(markdown)

    return markdown


def convert_url_with_jina(url: str, timeout: int = 30) -> str:
    """
    Convert URL to Markdown using Jina Reader API.

    Jina Reader handles JavaScript/SPA sites better than local parsing.
    """
    jina_url = f"https://r.jina.ai/{url}"

    try:
        result = subprocess.run(
            ["curl", "-s", "-m", str(timeout), jina_url],
            capture_output=True,
            text=True,
            timeout=timeout + 5
        )

        if result.returncode != 0:
            raise RuntimeError(f"curl failed with code {result.returncode}")

        content = result.stdout.strip()

        if not content:
            raise RuntimeError("Jina Reader returned empty content")

        return content

    except subprocess.TimeoutExpired:
        raise RuntimeError(f"Jina Reader timeout after {timeout}s")
    except FileNotFoundError:
        raise RuntimeError("curl not found, please install curl")


def convert_url_with_markitdown(url: str) -> str:
    """Convert URL using MarkItDown (fallback)."""
    md = MarkItDown()
    result = md.convert(url)
    return result.text_content


def convert_url(url: str, backend: str = "auto", timeout: int = 30) -> tuple:
    """
    Convert URL to Markdown.

    Args:
        url: Web URL to convert
        backend: "auto", "jina", or "markitdown"
        timeout: Timeout in seconds for Jina Reader

    Returns:
        Tuple of (markdown_content, backend_used)
    """
    if backend == "jina":
        return convert_url_with_jina(url, timeout), "jina"

    if backend == "markitdown":
        return convert_url_with_markitdown(url), "markitdown"

    # Auto mode: try Jina first, fallback to MarkItDown
    try:
        return convert_url_with_jina(url, timeout), "jina"
    except Exception as e:
        log_warning(f"Jina Reader failed: {e}, falling back to MarkItDown")
        return convert_url_with_markitdown(url), "markitdown"


def convert_pdf(file_path: Path, backend: str = "auto", **kwargs) -> tuple:
    """Convert PDF to Markdown. Returns (content, backend_used)."""
    from convert_pdf import convert_pdf as pdf_convert
    return pdf_convert(file_path, backend=backend, **kwargs)


def convert_image(file_path: Path, backend: str = "auto", **kwargs) -> tuple:
    """Convert image to text using OCR. Returns (content, backend_used)."""
    from convert_image import convert_image as img_convert
    content = img_convert(file_path, backend=backend, **kwargs)
    backend_used = backend if backend != "auto" else "tesseract"
    return content, backend_used


def convert_office(file_path: Path) -> tuple:
    """Convert Office files using MarkItDown."""
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content, "markitdown"


def convert_web(file_path: Path) -> tuple:
    """Convert local HTML files using MarkItDown."""
    md = MarkItDown()
    result = md.convert(str(file_path))
    return result.text_content, "markitdown"


def convert_text(file_path: Path) -> tuple:
    """Read text files directly with encoding detection."""
    # Try common encodings
    encodings = ["utf-8", "utf-8-sig", "big5", "gb2312", "latin-1"]

    for encoding in encodings:
        try:
            return file_path.read_text(encoding=encoding), "native"
        except UnicodeDecodeError:
            continue

    # Last resort: read with errors ignored
    return file_path.read_text(encoding="utf-8", errors="ignore"), "native"


def check_empty_content(markdown: str, file_type: str) -> list:
    """Check if content is empty or suspiciously short, return warnings."""
    warnings = []

    if not markdown or not markdown.strip():
        warnings.append("Output is empty - file may be scanned/image-based")
    elif len(markdown.strip()) < 50 and file_type == "pdf":
        warnings.append("Output is very short - consider using --pdf-backend marker for scanned PDFs")

    return warnings


def log_progress(current: int, total: int, filename: str, status: str = "converting"):
    """Print progress to stderr."""
    print(f"[{current}/{total}] {status}: {filename}", file=sys.stderr)


def discover_files(
    input_dir: Path,
    recursive: bool = False,
    extensions: Optional[set] = None
) -> List[Path]:
    """
    Discover all supported files in a directory.

    Args:
        input_dir: Directory to search
        recursive: Whether to search subdirectories
        extensions: Set of extensions to include (default: all supported)

    Returns:
        List of file paths sorted alphabetically
    """
    if extensions is None:
        extensions = ALL_SUPPORTED_EXTENSIONS

    files = []

    if recursive:
        for ext in extensions:
            files.extend(input_dir.rglob(f"*{ext}"))
    else:
        for ext in extensions:
            files.extend(input_dir.glob(f"*{ext}"))

    # Sort alphabetically for consistent ordering
    return sorted(files)


def convert_batch(
    input_dir: str,
    output_dir: Optional[str] = None,
    recursive: bool = False,
    parallel: int = 1,
    pdf_backend: str = "auto",
    ocr_backend: str = "auto",
    include_frontmatter: bool = False,
    clean_output: bool = True,
    output_format: str = "human",
    **kwargs
) -> dict:
    """
    Convert all supported files in a directory to Markdown.

    Args:
        input_dir: Input directory path
        output_dir: Output directory (default: same as input)
        recursive: Search subdirectories
        parallel: Number of parallel workers (1 = sequential)
        pdf_backend: PDF converter backend
        ocr_backend: OCR backend for images
        include_frontmatter: Add YAML frontmatter with metadata
        clean_output: Clean up markdown formatting
        output_format: "human" (readable) or "rag" (structured for LLM/RAG)
        **kwargs: Additional options passed to convert_document

    Returns:
        dict with batch statistics and per-file results
    """
    start_time = time.time()

    input_path = Path(input_dir)
    if not input_path.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    if not input_path.is_dir():
        raise ValueError(f"Not a directory: {input_dir}")

    # Set output directory
    if output_dir is None:
        out_path = input_path
    else:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

    # Discover files
    files = discover_files(input_path, recursive=recursive)

    if not files:
        return {
            "success": True,
            "total": 0,
            "converted": 0,
            "failed": 0,
            "skipped": 0,
            "results": [],
            "elapsed_seconds": 0.0,
        }

    total = len(files)
    results = []
    converted = 0
    failed = 0

    def process_file(file_path: Path, index: int) -> dict:
        """Process a single file and return result."""
        # Calculate relative path for output
        try:
            rel_path = file_path.relative_to(input_path)
        except ValueError:
            rel_path = file_path.name

        # Determine output path
        output_file = out_path / rel_path.with_suffix(".md")

        log_progress(index + 1, total, str(rel_path))

        try:
            result = convert_document(
                input_path=str(file_path),
                output_path=str(output_file),
                pdf_backend=pdf_backend,
                ocr_backend=ocr_backend,
                include_frontmatter=include_frontmatter,
                clean_output=clean_output,
                output_format=output_format,
                **kwargs
            )
            result["input_path"] = str(file_path)
            return result
        except Exception as e:
            return {
                "success": False,
                "input_path": str(file_path),
                "output_path": str(output_file),
                "error": str(e),
                "error_type": type(e).__name__,
            }

    # Process files (parallel or sequential)
    if parallel > 1:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(process_file, f, i): i
                for i, f in enumerate(files)
            }
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                if result.get("success"):
                    converted += 1
                else:
                    failed += 1
    else:
        for i, file_path in enumerate(files):
            result = process_file(file_path, i)
            results.append(result)
            if result.get("success"):
                converted += 1
            else:
                failed += 1

    # Sort results by input path for consistent output
    results.sort(key=lambda r: r.get("input_path", ""))

    elapsed_time = time.time() - start_time

    return {
        "success": failed == 0,
        "total": total,
        "converted": converted,
        "failed": failed,
        "input_dir": str(input_path),
        "output_dir": str(out_path),
        "results": results,
        "elapsed_seconds": round(elapsed_time, 2),
    }


def convert_document(
    input_path: str,
    output_path: Optional[str] = None,
    pdf_backend: str = "auto",
    ocr_backend: str = "auto",
    url_backend: str = "auto",
    url_timeout: int = 30,
    pages: Optional[str] = None,
    include_frontmatter: bool = False,
    clean_output: bool = True,
    output_format: str = "human",
    **kwargs
) -> dict:
    """
    Convert any supported document to Markdown.

    Args:
        input_path: Path to input file or URL
        output_path: Path for output (default: <input>.md, use "-" for stdout)
        pdf_backend: PDF converter backend
        ocr_backend: OCR backend for images
        url_backend: URL converter backend ("auto", "jina", "markitdown")
        url_timeout: Timeout for URL fetching
        pages: Page range for PDFs (e.g., "1-5", "1,3,5")
        include_frontmatter: Add YAML frontmatter with metadata
        clean_output: Clean up markdown formatting
        output_format: "human" (readable) or "rag" (structured for LLM/RAG)
        **kwargs: Additional options

    Returns:
        dict with success status, output_path, content (if stdout), and metadata
    """
    start_time = time.time()
    warnings = []
    backend_used = None
    metadata = {}
    output_to_stdout = output_path == "-"
    source_path = input_path  # Keep original for frontmatter

    # Handle URLs
    if is_url(input_path):
        if output_path is None:
            # Generate filename from URL
            parsed = urlparse(input_path)
            filename = parsed.path.split("/")[-1] or "index"
            if not filename.endswith(".md"):
                filename = filename.split(".")[0] + ".md"
            output_path = filename

        markdown, backend_used = convert_url(input_path, backend=url_backend, timeout=url_timeout)
        file_type = "url"
    else:
        # Handle local files
        input_file = Path(input_path)

        if not input_file.exists():
            raise FileNotFoundError(f"File not found: {input_file}")

        if output_path is None:
            output_path = str(input_file.with_suffix(".md"))

        file_type = detect_file_type(input_file)

        # Extract metadata for PDFs
        if file_type == "pdf" and include_frontmatter:
            metadata = extract_pdf_metadata(input_file)

        if file_type == "pdf":
            markdown, backend_used = convert_pdf(
                input_file,
                backend=pdf_backend,
                pages=pages,
                **kwargs
            )
        elif file_type == "image":
            markdown, backend_used = convert_image(input_file, backend=ocr_backend, **kwargs)
        elif file_type == "office":
            markdown, backend_used = convert_office(input_file)
        elif file_type == "web":
            markdown, backend_used = convert_web(input_file)
        elif file_type == "text":
            markdown, backend_used = convert_text(input_file)
        elif file_type == "unknown":
            try:
                md = MarkItDown()
                result = md.convert(str(input_file))
                markdown = result.text_content
                backend_used = "markitdown"
            except Exception as e:
                raise ValueError(f"Unsupported file type: {input_file.suffix}") from e
        else:
            raise ValueError(f"Unsupported file type: {file_type}")

    # Check for empty content
    warnings.extend(check_empty_content(markdown, file_type))

    # Clean up markdown
    if clean_output:
        markdown = cleanup_markdown(markdown)

    # Apply format-specific processing
    if output_format == "human":
        markdown = simplify_for_human(markdown)

    # Add frontmatter if requested
    if include_frontmatter and metadata:
        frontmatter = generate_frontmatter(metadata, source_path)
        markdown = frontmatter + markdown

    elapsed_time = time.time() - start_time

    result = {
        "success": True,
        "file_type": file_type,
        "backend_used": backend_used,
        "output_format": output_format,
        "content_length": len(markdown),
        "elapsed_seconds": round(elapsed_time, 2),
        "warnings": warnings,
    }

    # Add metadata if extracted
    if metadata:
        result["metadata"] = {k: v for k, v in metadata.items() if v}

    # Handle output
    if output_to_stdout:
        result["output_path"] = "-"
        result["content"] = markdown
    else:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(markdown, encoding="utf-8")
        result["output_path"] = str(output_file)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Convert documents to Markdown (PDF, Office, Images, HTML, URLs)"
    )

    # Input options (mutually exclusive: single file vs batch)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input", "-i",
        type=str,
        help="Input file path or URL"
    )
    input_group.add_argument(
        "--input-dir",
        type=str,
        help="Input directory for batch processing"
    )

    # Output options
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output Markdown file (default: <input>.md, use '-' for stdout)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for batch processing (default: same as input-dir)"
    )

    # Batch processing options
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="Recursively process subdirectories (batch mode)"
    )
    parser.add_argument(
        "--parallel", "-j",
        type=int,
        default=1,
        help="Number of parallel workers for batch processing (default: 1)"
    )
    parser.add_argument(
        "--pdf-backend",
        choices=["auto", "pymupdf4llm", "marker", "markitdown"],
        default="auto",
        help="PDF conversion backend (default: auto)"
    )
    parser.add_argument(
        "--ocr-backend",
        choices=["auto", "tesseract", "surya", "easyocr"],
        default="auto",
        help="OCR backend for images (default: auto)"
    )
    parser.add_argument(
        "--url-backend",
        choices=["auto", "jina", "markitdown"],
        default="auto",
        help="URL conversion backend (default: auto = jina with fallback)"
    )
    parser.add_argument(
        "--url-timeout",
        type=int,
        default=30,
        help="Timeout for URL fetching in seconds (default: 30)"
    )
    parser.add_argument(
        "--pages", "-p",
        type=str,
        help="Page range for PDFs (e.g., '1-5', '1,3,5', '1-3,7,10-12')"
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for image extraction from PDFs (default: 150)"
    )
    parser.add_argument(
        "--write-images",
        action="store_true",
        help="Extract and save images from PDFs"
    )
    parser.add_argument(
        "--frontmatter",
        action="store_true",
        help="Include YAML frontmatter with document metadata"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Disable markdown cleanup (keep original formatting)"
    )
    parser.add_argument(
        "--format", "-f",
        choices=["human", "rag"],
        default="human",
        help="Output format: 'human' (clean, readable) or 'rag' (preserves structure for LLM/RAG)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON (for agent integration)"
    )

    args = parser.parse_args()

    try:
        # Batch mode
        if args.input_dir:
            result = convert_batch(
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                recursive=args.recursive,
                parallel=args.parallel,
                pdf_backend=args.pdf_backend,
                ocr_backend=args.ocr_backend,
                include_frontmatter=args.frontmatter,
                clean_output=not args.no_cleanup,
                output_format=args.format,
                dpi=args.dpi,
                write_images=args.write_images,
            )

            # Print summary to stderr
            print(f"\nBatch complete: {result['converted']}/{result['total']} converted", file=sys.stderr)
            if result['failed'] > 0:
                print(f"Failed: {result['failed']}", file=sys.stderr)

            # Handle output
            if args.json:
                print(json.dumps(result, indent=2))
            else:
                # Print summary
                print(f"Converted: {result['converted']}/{result['total']}")
                if result['failed'] > 0:
                    print(f"Failed: {result['failed']}")
                    for r in result['results']:
                        if not r.get('success'):
                            print(f"  - {r['input_path']}: {r.get('error', 'Unknown error')}")

            return 0 if result['success'] else 1

        # Single file mode
        else:
            result = convert_document(
                input_path=args.input,
                output_path=args.output,
                pdf_backend=args.pdf_backend,
                ocr_backend=args.ocr_backend,
                url_backend=args.url_backend,
                url_timeout=args.url_timeout,
                pages=args.pages,
                dpi=args.dpi,
                write_images=args.write_images,
                include_frontmatter=args.frontmatter,
                clean_output=not args.no_cleanup,
                output_format=args.format,
            )

            # Print warnings to stderr
            for warning in result.get("warnings", []):
                log_warning(warning)

            # Handle output
            if args.output == "-" and not args.json:
                # Direct stdout output
                print(result["content"])
            elif args.json:
                # JSON output (remove content for file output to keep JSON clean)
                if "content" in result and args.output != "-":
                    del result["content"]
                print(json.dumps(result, indent=2))
            else:
                print(result["output_path"])

            return 0

    except FileNotFoundError as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e), "error_type": "FileNotFoundError"}))
        else:
            log_error(str(e))
        return 1

    except ValueError as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e), "error_type": "ValueError"}))
        else:
            log_error(str(e))
        return 1

    except Exception as e:
        if args.json:
            print(json.dumps({"success": False, "error": str(e), "error_type": type(e).__name__}))
        else:
            log_error(str(e))
        return 1


if __name__ == "__main__":
    sys.exit(main())
