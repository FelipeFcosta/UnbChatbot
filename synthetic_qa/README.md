# Synthetic QA Generator for Institutional Chatbots

A tool to generate high-quality question-answer pairs from institutional websites, academic regulations, and educational content for training specialized chatbots.

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Basic Usage](#basic-usage)
7. [Processing Different Document Types](#processing-different-document-types)
8. [FAQ Processing](#faq-processing)
9. [Optimizing for Quality](#optimizing-for-quality)
10. [Output and Integration](#output-and-integration)
11. [Troubleshooting](#troubleshooting)

## Introduction

The Synthetic QA Generator extracts content from institutional documents, preserves their structure, and uses Large Language Models (LLMs) to generate contextually relevant questions and answers. This tool is especially designed for universities looking to build domain-specific chatbots with factual accuracy and comprehensive coverage of educational content.

Key benefits:
- Maintains document context and factual accuracy
- Specializes in handling educational content and FAQ documents
- Supports multiple LLM providers (OpenAI and Google Gemini)
- Creates contextually relevant QA pairs that reflect institutional knowledge

## System Overview

The system consists of several components:

1. **File Processor**: Extracts text from HTML, PDF, and text files
2. **Text Chunker**: Segments text while preserving context
3. **FAQ Processor**: Specialized handling for FAQ documents
4. **QA Generator**: Creates questions and answers from document content
5. **LLM Client**: Interfaces with language models

## Prerequisites

- Python 3.8 or newer
- LLM API access (OpenAI or Google Gemini)
- Input documents (website clone, PDFs, text files)

Required packages:
```
beautifulsoup4, PyMuPDF, openai, google-genai, yaml, tqdm, requests, html5lib, python-slugify
```

## Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/synthetic-qa-generator.git
cd synthetic-qa-generator
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Set Up API Keys

For OpenAI:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Google Gemini:
```bash
export GEMINI_API_KEY="your-api-key-here"
```

## Configuration

Create a configuration file `config.yaml`:

```yaml
# Factual accuracy mode
factual_mode: true

# Processing settings for specific file types
processing:
  html_pages:
    full_document_threshold: 20000
    comprehensive_questions: true
  
  faq:
    enabled: true
    generate_rephrased_questions: true
    generate_related_questions: true
    max_rephrased_questions: 7
    max_related_questions: 3
    context_size: 3

# API providers configuration
providers:
  faq_extraction:
    provider: genai  # or openai
    model: gemini-2.0-flash  # or gpt-3.5-turbo, gpt-4
    temperature: 0.4
    max_tokens: 8192
    
  question:
    provider: genai  # or openai
    model: gemini-2.0-flash  # or gpt-3.5-turbo, gpt-4
    temperature: 0.4
    max_tokens: 8192
  
  answer:
    provider: genai  # or openai
    model: gemini-2.0-flash  # or gpt-3.5-turbo, gpt-4
    temperature: 0.5
    max_tokens: 8192

# File processing settings
file_processing:
  include_extensions:
    - .html
    - .htm
    - .pdf
    - .txt
    - .md
```

### Key Configuration Options

- `factual_mode`: Preserves document context for factual accuracy
- `processing.html_pages`: Controls HTML processing behavior
- `processing.faq`: Settings for FAQ document handling
- `providers`: Configure different LLM providers for various generation tasks
- `file_processing`: Control which files are included

## Basic Usage

```bash
python synthetic_qa_generator.py --input /path/to/documents --output ./output
```

### Command Line Arguments

```
--config CONFIG     Path to configuration file (default: synthetic_qa/config.yaml)
--input INPUT       Directory containing input files (required)
--output OUTPUT     Directory to save output files (default: ./output)
--threads THREADS   Maximum number of concurrent workers (default: 4)
--factual           Enable factual mode (overrides config)
```

## Processing Different Document Types

### HTML Documents

HTML files receive specialized processing:
- Structure preservation with semantic heading levels
- Link conversion to markdown format
- Navigation and non-content elements removal
- Comprehensive question coverage for smaller documents

### PDF Documents

PDF files are processed with PyMuPDF:
- Text extraction with flow preservation
- Structured content representation
- Chunking based on semantic boundaries

### FAQ Documents

The system automatically detects FAQs through:
- Filename and title indicators ("faq", "perguntas frequentes", etc.)
- Structural patterns (details/summary tags, bold text followed by paragraphs)
- Question patterns within headings and content

## FAQ Processing

FAQ documents receive enhanced processing:

1. **Automatic extraction** of existing QA pairs
2. **Style variations** to create different phrasings of the same question
3. **Related question generation** for comprehensive coverage
4. **Context preservation** to maintain topical relationships

Benefits:
- Multiplies effective training examples from each original QA pair
- Captures different ways users might ask the same question
- Preserves institutional terminology and domain context

## Optimizing for Quality

### LLM Provider Selection

The system supports multiple LLM providers:

1. **OpenAI**: 
   - Models: gpt-3.5-turbo, gpt-4, etc.
   - Higher quality with GPT-4
   - Better for English content

2. **Google Gemini**:
   - Models: gemini-2.0-flash, etc.
   - Good performance/cost balance
   - Strong multilingual capabilities

### Temperature Settings

- **Questions**: 0.4-0.7 (higher for more diverse questions)
- **Answers**: 0.2-0.5 (lower for more factual accuracy)

### Chunking Parameters

Adjust in the global configuration:
```yaml
global:
  max_chunk_size: 4000
  min_chunk_size: 1000
  overlap_size: 200
```

## Output and Integration

Output is generated in JSON format suitable for:
- LLM fine-tuning
- RAG system training data
- Embedding model training

Example output structure:
```json
[
  {
    "question": "What are the laboratory facilities at the Electrical Engineering Department?",
    "answer": "According to information from www.ene.unb.br, the Electrical Engineering Department has several laboratories including...",
    "source": "ene.unb.br/facilities.html",
    "url": "www.ene.unb.br/facilities",
    "domain": "ene.unb.br",
    "institution": "Departamento de Engenharia Elétrica",
    "type": "standard"
  },
  ...
]
```

## Troubleshooting

### Common Issues

1. **API Rate Limiting**:
   - Reduce the number of concurrent workers with `--threads`
   - The system implements exponential backoff automatically

2. **Memory Usage**:
   - Processing large documents may require more memory
   - Reduce `max_chunk_size` or process fewer files simultaneously

3. **Quality Issues**:
   - Try using a more capable model (GPT-4 or Gemini Pro)
   - Adjust temperature settings
   - Check debug outputs in the `debug` directory