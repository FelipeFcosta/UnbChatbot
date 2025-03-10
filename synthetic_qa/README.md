# Comprehensive Guide: Synthetic QA Generator for Institutional Chatbots

This guide provides complete instructions for setting up, configuring, and using the Synthetic QA Generator to create high-quality training data for institutional chatbots. The system is specifically optimized for university websites, academic regulations, and institutional content.

## Table of Contents

1. [Introduction](#introduction)
2. [System Overview](#system-overview)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Basic Usage](#basic-usage)
7. [Advanced Features](#advanced-features)
8. [Processing Different Document Types](#processing-different-document-types)
9. [Optimizing for Quality](#optimizing-for-quality)
10. [Output and Integration](#output-and-integration)
11. [Troubleshooting](#troubleshooting)
12. [Best Practices](#best-practices)
13. [FAQs](#faqs)

## Introduction

The Synthetic QA Generator is a tool designed to create question-answer pairs from institutional content for training specialized chatbots. It processes various document types (HTML, PDF, text), extracts content while preserving structure, and uses Large Language Models (LLMs) to generate contextually relevant questions and answers.

Key benefits:
- Generates training data that preserves factual accuracy
- Maintains document context and structure
- Handles different document types with specialized processing
- Configurable to optimize for your specific needs
- Creates comprehensive coverage of document content

## System Overview

The system consists of several key components:

1. **File Processor**: Extracts text from different file formats (HTML, PDF, TXT)
2. **Text Chunker**: Segments text into manageable pieces (or keeps it intact for factual accuracy)
3. **Question Generator**: Creates relevant questions based on document content
4. **Answer Generator**: Produces accurate answers to those questions
5. **Output Manager**: Organizes generated QA pairs into usable formats

![System Architecture](https://example.com/system-architecture.png)

## Prerequisites

Before installation, ensure you have:

- Python 3.8 or newer
- An OpenAI API key (or access to another supported LLM)
- Adequate disk space for processing and storing documents
- Input documents (website clone, PDFs, text files)

Required Python packages:
- `beautifulsoup4` (HTML processing)
- `PyMuPDF` (PDF processing)
- `openai` (LLM API access)
- `yaml` (configuration)
- `tqdm` (progress tracking)
- `requests` (HTTP requests)

## Installation

### Step 1: Clone or Download the Repository

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

## Configuration

The system is configured using a YAML file. Create a file named `synthetic_qa_config.yaml` with the following structure:

```yaml
# Factual accuracy mode 
factual_mode: true  # Set to false for standard generation mode

# Processing settings for specific file types
processing:
  # HTML page specific settings
  html_pages:
    # Preserve heading structure when extracting text
    preserve_structure: true
    # Character threshold below which HTML documents are kept as a single unit
    full_document_threshold: 20000
    # Generate comprehensive questions covering all sections
    comprehensive_questions: true

# API providers configuration
providers:
  # Configuration for question generation
  question:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.7
    max_tokens: 500
  
  # Configuration for answer generation
  answer:
    provider: openai
    model: gpt-3.5-turbo
    temperature: 0.3
    max_tokens: 1000

# File processing settings
file_processing:
  # File types to include
  include_extensions:
    - .html
    - .htm
    - .pdf
    - .txt
    - .md
  
  # Directories or files to exclude
  exclude_patterns:
    - images/
    - assets/
    - .git/
    - styles/
    - js/
```

### Key Configuration Options

#### Factual Mode

The `factual_mode` setting prioritizes factual accuracy by:
- Preserving full document context when possible
- Using specialized prompts that emphasize factual information
- Applying lower temperature settings for more consistent outputs

```yaml
factual_mode: true  # Recommended for institutional content
```

#### HTML Processing

HTML-specific settings control structure preservation and chunking behavior:

```yaml
processing:
  html_pages:
    preserve_structure: true        # Maintains headings and structure
    full_document_threshold: 20000  # Pages under this size kept whole
    comprehensive_questions: true   # Questions cover all sections
```

#### LLM Settings

Configure the question and answer generation separately:

```yaml
providers:
  question:
    provider: openai
    model: gpt-3.5-turbo     # Or gpt-4 for higher quality
    temperature: 0.7         # Higher for more varied questions
    max_tokens: 500
  
  answer:
    provider: openai  
    model: gpt-3.5-turbo     # Or gpt-4 for higher quality
    temperature: 0.3         # Lower for more factual answers
    max_tokens: 1000         # Higher for more detailed answers
```

## Basic Usage

### Running the Generator

The basic command to run the tool is:

```bash
python synthetic_qa_generator.py --input /path/to/documents --output ./output
```

### Command Line Options

```
--config CONFIG     Path to configuration YAML file (default: synthetic_qa_config.yaml)
--input INPUT       Directory containing input files (required)
--output OUTPUT     Directory to save output files (default: ./output)
--threads THREADS   Maximum number of concurrent workers (default: 4)
--factual           Enable factual mode (overrides config file)
```

### Example Usage

Process a website clone with factual mode:

```bash
python synthetic_qa_generator.py --input ./website_clone --output ./qa_data --factual
```

Process a directory of PDFs with 8 concurrent threads:

```bash
python synthetic_qa_generator.py --input ./pdf_documents --output ./qa_data --threads 8
```

## Advanced Features

### Document Grouping

Related files are automatically grouped for processing together, preserving cross-document context. This is especially helpful for content that spans multiple files but forms a coherent topic.

### Structure Preservation

The system preserves document structure like headings, sections, and paragraphs:

```
# Major Heading
Content under major heading

## Subheading
Content under subheading
```

This helps generate more structured and contextually aware questions.

### Comprehensive Question Coverage

For HTML pages processed as single documents, the system generates questions that cover all sections of the page, ensuring comprehensive coverage.

### Caching and Incremental Processing

The system caches generated questions and answers to avoid redundant work when processing files incrementally.

## Processing Different Document Types

### HTML Files

HTML files receive special treatment:
- Structure preservation (headings, sections)
- Small files kept intact as single documents
- Comprehensive question generation covering all sections
- Navigation, scripts, and other non-content elements removed

### PDFs

PDF processing:
- Text extraction using PyMuPDF
- Page structure preserved where possible
- Text flow reconstructed for better context

### Text-Based Files

For TXT, MD, and other text formats:
- Basic structure preserved
- Chunking based on semantic boundaries when needed

# Processing FAQ Documents

The Synthetic QA Generator now includes specialized handling for FAQ documents, significantly enhancing the quality and coverage of training data generated from institutional FAQ content.

## Automatic FAQ Detection

The system uses several signals to automatically identify FAQ documents:

- Filenames containing terms like "faq", "perguntas", "frequentes", "duvidas", "q&a"
- Page titles containing similar FAQ indicator terms
- HTML structure using details/summary tags (common accordion-style FAQs)
- Pattern of bold/strong text followed by paragraphs (common in manually formatted FAQs)
- HTML definition lists (dt/dd tags)
- Headings followed by paragraphs where the heading ends with a question mark

## Enhanced FAQ Processing

For documents identified as FAQs, the tool:

1. **Extracts existing QA pairs** from various HTML structures
2. **Preserves contextual relationships** between neighboring questions
3. **Generates rephrased variations** of original questions
4. **Creates related questions** that can be answered by the same content

This approach significantly enhances training data by:

- Capturing different ways users might phrase the same question
- Extracting implicit information embedded in answers 
- Maintaining the natural flow and organization of the FAQ
- Multiplying the effective training examples from each original QA pair

## Example

**Original FAQ question:**
```
What are the enrollment requirements for the Computer Science program?
```

**Rephrased variations:**
```
What do I need to enroll in Computer Science?
How can I get into the Computer Science program?
What are the prerequisites for Computer Science enrollment?
What criteria must I meet to join the Computer Science program?
```

**Related questions answerable by the same content:**
```
When is the deadline for Computer Science enrollment?
Is there a minimum grade requirement for the Computer Science program?
Do I need to submit recommendation letters for Computer Science admission?
```

## Usage Scenario

This specialized FAQ processing is particularly valuable for institutional websites that contain:

- Dedicated FAQ pages
- Course information with Q&A sections
- Administrative procedures explained as questions and answers
- Student services pages with common questions

The system will automatically detect these patterns and apply enhanced processing.

## Configuration

Configure FAQ processing behavior using these settings in your configuration file:

```yaml
processing:
  faq:
    enabled: true
    generate_rephrased_questions: true
    generate_related_questions: true
    max_rephrased_questions: 5
    max_related_questions: 3
    context_size: 2
```

- `enabled`: Turn FAQ processing on/off
- `generate_rephrased_questions`: Generate variations of the original question
- `generate_related_questions`: Generate new questions answerable by the same content
- `max_rephrased_questions`: Limit how many rephrased questions to keep (0 = unlimited)
- `max_related_questions`: Limit how many related questions to keep (0 = unlimited)
- `context_size`: Number of neighboring QA pairs to include as context

## Impact on Training Data Volume

The FAQ enhancements significantly multiply the effective size of your training dataset. For example:

- A single FAQ with 10 QA pairs
- With 5 rephrased questions per original question
- And 3 related questions per original question
- Results in up to 90 total QA pairs (10 original + 50 rephrased + 30 related)

This creates a much richer dataset that improves the chatbot's ability to understand varied phrasings and implicit questions.

## Optimizing for Quality

### Balancing Chunk Size and Context

Larger chunks preserve more context but may hit model token limits. Configure the chunking parameters:

```yaml
global:
  max_chunk_size: 4000    # Increase for more context
  min_chunk_size: 1000    # Minimum to ensure meaningful content
  overlap_size: 200       # Overlap between chunks
```

### Temperature Settings

Temperature controls randomness and creativity:
- Lower (0.1-0.3): More consistent, factual results
- Higher (0.7-0.9): More creative, varied questions

For institutional chatbots, use:
- Question generation: 0.5-0.7
- Answer generation: 0.2-0.4

### Model Selection

Model capabilities significantly impact quality:
- `gpt-3.5-turbo`: Good balance of cost and quality
- `gpt-4`: Higher quality but more expensive
- `gpt-4-turbo`: Largest context window for full documents

## Output and Integration

### Output Structure

The tool generates:
1. JSON file with all QA pairs:
   ```
   output/synthetic_qa_data.json
   ```

2. Individual QA files for inspection:
   ```
   output/qa_generation_output/qa_pairs/
   ```

3. Debug information:
   ```
   output/qa_generation_output/debug/
   ```

### JSON Format

The output JSON format is:

```json
[
  {
    "question": "What laboratories are available in the Department of Electrical Engineering?",
    "answer": "According to the information provided, the Department of Electrical Engineering has several laboratories including: Laboratory of Communication Principles, Laboratory of Electrical Circuits, Laboratory of Digital Techniques, Laboratory of Control, Laboratory of Industrial Process Control, Laboratory of Energy Conversion, Laboratory of Electromagnetism and Antennas, Laboratory of Electrical Installations and Basic Electricity, and Laboratory of Electrical and Magnetic Materials. The department has the largest quantity and diversification of laboratories at the University of Brasília.",
    "source": "ene.unb.br/index.php/institucional/infraestrutura.html",
    "chunk_hash": "75a8b3c219fe4d78_0"
  },
  // Additional QA pairs...
]
```

### Integration with Training Pipelines

The output JSON can be:
1. Used directly for fine-tuning LLMs
2. Converted to JSONL format for specific fine-tuning APIs
3. Processed further for RAG systems
4. Used as training data for embedding models

## Troubleshooting

### Common Issues

#### API Rate Limiting

**Problem**: OpenAI API rate limit errors
**Solution**: Reduce the `--threads` parameter and implement exponential backoff (already included in the code)

#### Out of Memory

**Problem**: Processing large documents causes memory issues
**Solution**: Reduce `max_chunk_size` or process fewer files simultaneously with lower `--threads`

#### Poor Quality Questions

**Problem**: Generated questions are too generic
**Solution**: 
- Try a more capable model (`gpt-4`)
- Adjust temperature (higher for more creative questions)
- Check if document chunks are maintaining enough context

### Debug Information

For debugging, check the files in:
```
output/qa_generation_output/debug/
```

These files contain the prompts sent to the LLM and can help diagnose quality issues.

## Best Practices

1. **Start small**: Begin with a subset of your documents to calibrate settings

2. **Review regularly**: Manually review a sample of generated QA pairs

3. **Iterate on configuration**: Adjust settings based on your specific content types

4. **Organize input data**: Group related content in the same directories for better context

5. **Balance coverage vs. quality**: 
   - More documents → broader coverage
   - More time per document → higher quality
   - Configure accordingly based on your priorities

6. **Document Content Preparation**:
   - Clean up HTML where possible
   - Ensure documents have clear headings and structure
   - Remove duplicative content before processing

## FAQs

**Q: How many QA pairs will be generated per document?**
A: It depends on document size and complexity. For factual mode with comprehensive questions enabled, HTML pages processed as single documents generally yield 6-10 questions. Documents split into chunks typically yield 3-5 questions per chunk.

**Q: How much does it cost to run this with OpenAI?**
A: Cost varies based on document volume and model selection. Processing 100 pages with GPT-3.5-Turbo typically costs $1-5 USD. Using GPT-4 will increase costs substantially but may improve quality.

**Q: Can I use other LLM providers?**
A: The system is designed to be extensible. Currently, OpenAI is fully implemented, but you can extend the `LLMClient` class to support other providers.

**Q: How do I optimize for multilingual content?**
A: Ensure your documents are properly encoded (UTF-8), and consider using a model with strong multilingual capabilities (like GPT-4). The system will preserve the original language in extraction.

**Q: What's the ideal document size for best results?**
A: Documents that fit within the model's context window as a single unit (typically under 20,000 characters for GPT-3.5-Turbo) yield the best results because they maintain full context.

---
