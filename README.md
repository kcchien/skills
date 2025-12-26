# Skills

> **Note:** This repository contains custom skills for Claude Code. For information about the Agent Skills standard, see [agentskills.io](http://agentskills.io).

## What are Skills?

Skills are folders of instructions, scripts, and resources that Claude loads dynamically to improve performance on specialized tasks. They teach Claude how to complete specific tasks in a repeatable way.

**Related Resources:**
- [What are skills?](https://support.claude.com/en/articles/12512176-what-are-skills)
- [Using skills in Claude](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [How to create custom skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)

## About This Repository

This repository contains custom skills for document processing and conversion tasks. Each skill is self-contained in its own folder with a `SKILL.md` file containing instructions and metadata.

## Available Skills

### document-to-markdown

Convert documents and URLs to clean Markdown for LLM/RAG use.

**Supported Formats:**
| Type | Formats |
|------|---------|
| Documents | PDF, DOCX, PPTX, XLSX |
| Images | PNG, JPG, JPEG, WEBP, TIFF (with OCR) |
| Web | HTML, URLs |
| Text | TXT, MD, CSV, JSON, XML |

**Key Features:**
- Multiple backends (pymupdf4llm for speed, marker for scanned docs)
- Batch processing with parallel execution
- RAG-optimized output format
- Structured JSON output for agent integration

## Installation

### Using Skills CLI (Recommended)

```bash
# Install skills-cli first
pip install git+https://github.com/kcchien/skills-cli.git

# List available skills from this repo
skills-cli list --repo https://github.com/kcchien/skills

# Install document-to-markdown skill
skills-cli install --repo https://github.com/kcchien/skills --skills document-to-markdown
```

### Manual Installation

```bash
git clone https://github.com/kcchien/skills.git
cp -r skills/document-to-markdown ~/.claude/skills/
```

That's it! Claude will automatically discover the skill and handle dependencies when needed.

## License

MIT License

---

# 中文說明

> **備註：** 本儲存庫包含 Claude Code 的自訂技能。關於 Agent Skills 標準，請參閱 [agentskills.io](http://agentskills.io)。

## 什麼是 Skills？

Skills 是包含指令、腳本和資源的資料夾，Claude 會動態載入以提升特定任務的表現。它們教導 Claude 如何以可重複的方式完成特定任務。

**相關資源：**
- [什麼是 Skills？](https://support.claude.com/en/articles/12512176-what-are-skills)
- [在 Claude 中使用 Skills](https://support.claude.com/en/articles/12512180-using-skills-in-claude)
- [如何建立自訂 Skills](https://support.claude.com/en/articles/12512198-creating-custom-skills)

## 關於本儲存庫

本儲存庫包含文件處理和轉換任務的自訂技能。每個技能都獨立存放於各自的資料夾中，並包含 `SKILL.md` 檔案描述指令和中繼資料。

## 可用技能

### document-to-markdown

將文件和網址轉換為乾淨的 Markdown 格式，適用於 LLM/RAG。

**支援格式：**
| 類型 | 格式 |
|------|------|
| 文件 | PDF、DOCX、PPTX、XLSX |
| 圖片 | PNG、JPG、JPEG、WEBP、TIFF（支援 OCR） |
| 網頁 | HTML、URLs |
| 文字 | TXT、MD、CSV、JSON、XML |

**主要特色：**
- 多後端支援（pymupdf4llm 快速處理、marker 適合掃描文件）
- 批次處理支援平行執行
- RAG 最佳化輸出格式
- 結構化 JSON 輸出方便 Agent 整合

## 安裝方式

### 使用 Skills CLI（推薦）

```bash
# 先安裝 skills-cli
pip install git+https://github.com/kcchien/skills-cli.git

# 列出此 repo 的可用技能
skills-cli list --repo https://github.com/kcchien/skills

# 安裝 document-to-markdown 技能
skills-cli install --repo https://github.com/kcchien/skills --skills document-to-markdown
```

### 手動安裝

```bash
git clone https://github.com/kcchien/skills.git
cp -r skills/document-to-markdown ~/.claude/skills/
```

完成！Claude 會自動探索此技能，並在需要時處理相依套件。

## 授權

MIT License
