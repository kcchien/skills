# Troubleshooting Guide

Common issues and solutions for document-to-markdown conversion.

## PDF Issues

### Output is empty or very short

**Symptoms:**
- JSON output shows `"content_length": 0` or very small number
- Warning: "Output is empty - file may be scanned/image-based"

**Causes:**
- PDF is scanned (image-based)
- PDF uses custom fonts that can't be extracted
- PDF is encrypted or protected

**Solutions:**
```bash
# Use Marker backend for scanned PDFs
python scripts/gateway.py --input scanned.pdf --pdf-backend marker

# If Marker not installed
pip install marker-pdf
```

### Tables are malformed

**Symptoms:**
- Table columns don't align
- Data appears as continuous text

**Note:** v0.7.0+ auto-detects and fixes simple comparison tables (with ✓/x patterns).

**Solutions:**
```bash
# For complex tables, use Marker backend
python scripts/gateway.py --input tables.pdf --pdf-backend marker

# Disable auto-cleanup if table fix causes issues
python scripts/gateway.py --input doc.pdf --no-cleanup
```

### Multi-column layout issues

**Symptoms:**
- Text from different columns is merged
- Reading order is wrong

**Solutions:**
```bash
# Marker handles multi-column better
python scripts/gateway.py --input multicolumn.pdf --pdf-backend marker
```

---

## Image OCR Issues

### No OCR backend available

**Error:** `RuntimeError: No OCR backend available`

**Solution:**
```bash
# Install Tesseract
brew install tesseract
pip install pytesseract
```

### Wrong language detected

**Symptoms:**
- Garbled characters in output
- Missing characters

**Solution:**
```bash
# Specify correct language
python scripts/gateway.py --input chinese.png --ocr-backend tesseract --lang chi_tra

# For Japanese
python scripts/gateway.py --input japanese.png --lang jpn
```

### Poor OCR quality

**Causes:**
- Low resolution image
- Poor contrast
- Skewed/rotated text

**Solutions:**
1. Pre-process image (increase contrast, straighten)
2. Use higher DPI source if available
3. Try different backend: `--ocr-backend easyocr`

---

## URL Conversion Issues

### Jina Reader timeout

**Error:** `RuntimeError: Jina Reader timeout after 30s`

**Solutions:**
```bash
# Increase timeout
python scripts/gateway.py --input https://slow-site.com --url-timeout 60

# Use local backend (no JS support)
python scripts/gateway.py --input https://site.com --url-backend markitdown
```

### Rate limit exceeded

**Symptoms:**
- Errors after many requests
- HTTP 429 responses

**Solutions:**
- Wait 1 minute between batches
- Get free Jina API key for higher limits (500 RPM)
- Use `--url-backend markitdown` for static sites

### Content not extracted

**Symptoms:**
- Output contains only navigation/footer
- Main content missing

**Causes:**
- Site blocks Jina Reader
- Content loaded via complex JavaScript

**Solutions:**
```bash
# Try MarkItDown as fallback
python scripts/gateway.py --input https://site.com --url-backend markitdown
```

---

## General Issues

### Unsupported file type

**Error:** `ValueError: Unsupported file type: .xyz`

**Supported formats:**
- PDF: `.pdf`
- Office: `.docx`, `.doc`, `.pptx`, `.ppt`, `.xlsx`, `.xls`
- Images: `.png`, `.jpg`, `.jpeg`, `.webp`, `.tiff`, `.bmp`, `.gif`
- Web: `.html`, `.htm`
- Text: `.txt`, `.md`, `.rst`, `.csv`, `.json`, `.xml`
- URLs: `http://`, `https://`

### Encoding issues

**Symptoms:**
- Garbled characters in output
- UnicodeDecodeError

**Note:** The tool auto-detects encoding (UTF-8, BIG5, GB2312, Latin-1). If issues persist, convert source file to UTF-8 first.

### Out of memory

**Symptoms:**
- Process killed
- MemoryError

**Solutions:**
```bash
# For large PDFs, process specific pages
python scripts/gateway.py --input huge.pdf --pages 1-50

# Then continue with remaining pages
python scripts/gateway.py --input huge.pdf --pages 51-100 --output huge_part2.md
```

---

## JSON Output Parsing

### Success response
```json
{
  "success": true,
  "output_path": "/path/to/output.md",
  "file_type": "pdf",
  "backend_used": "pymupdf4llm",
  "content_length": 15432,
  "elapsed_seconds": 1.23,
  "warnings": []
}
```

### Error response
```json
{
  "success": false,
  "error": "File not found: /path/to/file.pdf",
  "error_type": "FileNotFoundError"
}
```

### Checking for issues
```bash
# Parse JSON and check warnings
result=$(python scripts/gateway.py --input doc.pdf --json)
echo "$result" | jq '.warnings'

# Check if successful
echo "$result" | jq '.success'
```
