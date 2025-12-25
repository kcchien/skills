# Backend Reference

Detailed information about each conversion backend.

## PDF Backends

### PyMuPDF4LLM (default)

The primary PDF backend, optimized for text-based PDFs.

| Attribute | Value |
|-----------|-------|
| Speed | Fast (~1-2 seconds for typical documents) |
| Model Size | None (no download required) |
| Best For | Text-based PDFs, reports, articles |
| Limitations | Scanned PDFs produce empty output |

**Options:**
- `--dpi`: Image extraction DPI (default: 150)
- `--write-images`: Extract embedded images
- `--pages`: Page range (e.g., `1-10`, `1,3,5`)

### Marker

Deep learning-based converter for complex documents.

| Attribute | Value |
|-----------|-------|
| Speed | Medium (~5-15 seconds) |
| Model Size | ~1.3GB (first run downloads models) |
| Best For | Scanned PDFs, complex tables, multi-column layouts |
| Limitations | Slower, requires more memory |

**When to use:**
- PDF output is empty or very short
- Document has complex tables or figures
- Document is scanned/image-based
- Multi-column layouts

**Installation:**
```bash
pip install marker-pdf
```

### MarkItDown

Fallback option using Microsoft's converter.

| Attribute | Value |
|-----------|-------|
| Speed | Fast |
| Model Size | None |
| Best For | Simple PDFs, fallback |
| Limitations | Less structure preservation |

---

## OCR Backends (Images)

### Tesseract (default)

Traditional OCR engine, widely used and reliable.

| Attribute | Value |
|-----------|-------|
| Speed | Fast |
| Model Size | ~50MB per language |
| Best For | Clear images, printed text |
| Limitations | Struggles with handwriting |

**Installation:**
```bash
brew install tesseract
pip install pytesseract
```

**Language support:**
```bash
# List available languages
tesseract --list-langs

# Common language codes
--lang eng      # English (default)
--lang chi_tra  # Traditional Chinese
--lang chi_sim  # Simplified Chinese
--lang jpn      # Japanese
--lang kor      # Korean
--lang deu      # German
--lang fra      # French
```

### Surya

Deep learning OCR that comes with Marker.

| Attribute | Value |
|-----------|-------|
| Speed | Medium |
| Model Size | Included with Marker |
| Best For | Complex layouts, mixed languages |

### EasyOCR

Deep learning based, good multilingual support.

| Attribute | Value |
|-----------|-------|
| Speed | Medium |
| Model Size | ~100MB+ per language |
| Best For | Multiple languages in one image |
| Default Languages | English + Traditional Chinese |

**Installation:**
```bash
pip install easyocr
```

---

## URL Backends

### Jina Reader (default)

Cloud-based converter that handles JavaScript/SPA sites.

| Attribute | Value |
|-----------|-------|
| Speed | ~3-10 seconds |
| Rate Limit | 20 RPM (free), 500 RPM (with API key) |
| Best For | Modern websites, SPAs, dynamic content |
| Limitations | Requires internet, rate limited |

**Features:**
- Headless Chrome rendering
- Readability content extraction
- Removes navigation, ads, footers
- Handles JavaScript-rendered content

**Options:**
- `--url-timeout`: Timeout in seconds (default: 30)

### MarkItDown (fallback)

Local HTML parsing, no JavaScript support.

| Attribute | Value |
|-----------|-------|
| Speed | Fast |
| Best For | Static HTML, local files |
| Limitations | No JavaScript rendering |

---

## Backend Selection Logic

### PDF (auto mode)
```
1. Try PyMuPDF4LLM
2. If output < 100 chars → try Marker (if installed)
3. If Marker fails → try MarkItDown
4. If all fail → raise error
```

### Image (auto mode)
```
1. Try Tesseract
2. If fails → try Surya
3. If fails → try EasyOCR
4. If all fail → raise error
```

### URL (auto mode)
```
1. Try Jina Reader
2. If fails/timeout → fallback to MarkItDown
```
