# docling_ocr

[![PyPI version](https://badge.fury.io/py/docling_ocr.svg)](https://badge.fury.io/py/docling_ocr)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/release/python-370/)

A powerful Python package for extracting text from images and documents using advanced LLM-based models.

## Overview

`docling_ocr` leverages state-of-the-art language models specifically designed for document understanding tasks. Unlike traditional OCR engines that rely solely on character recognition, `docling_ocr` uses language models that understand document context, layouts, and can handle various document formats with high accuracy.

Built on top of models like SmolDocling, this package provides a simple, intuitive interface for document text extraction tasks.

## Features

- **LLM-powered extraction**: Uses advanced language models trained specifically for document understanding
- **Context-aware recognition**: Understands document layouts and context for improved accuracy
- **Multi-format support**: Works with scanned documents, forms, receipts, and other text-heavy images
- **Simple API**: Easy-to-use interface with both file and image object inputs
- **Batch processing**: Process entire directories of documents efficiently
- **Flexible output options**: Return text or save directly to files
- **Extensible architecture**: Abstract base class makes it easy to add new models

## Installation

```bash
pip install docling_ocr
```

### Requirements

- Python 3.7+
- PyTorch 1.10.0+
- Transformers 4.15.0+
- Pillow 8.0.0+

## Quick Start

### Basic Usage

```python
from docling_ocr import SmolDoclingExtractor

# Initialize the extractor
extractor = SmolDoclingExtractor()

# Extract text from an image file
text = extractor.extract_text("path/to/document.jpg")
print(text)

# Or use the shorthand callable interface
text = extractor("path/to/document.jpg")
```

### Using with PIL Images

```python
from docling_ocr import SmolDoclingExtractor
from PIL import Image

# Initialize the extractor
extractor = SmolDoclingExtractor()

# Open image with PIL
image = Image.open("path/to/document.jpg")

# Extract text
text = extractor.extract_text_from_image(image)
print(text)
```

### Batch Processing

```python
from docling_ocr import SmolDoclingExtractor
from docling_ocr.utils import batch_process

# Initialize extractor
extractor = SmolDoclingExtractor()

# Process all images in a directory
results = batch_process(
    extractor, 
    image_dir="path/to/documents/", 
    output_dir="path/to/output/",
    extensions=['.jpg', '.png', '.pdf']  # Optional: specify file extensions
)

# Results contains a dictionary mapping filenames to extracted text
for filename, text in results.items():
    print(f"File: {filename}")
    print(f"Text: {text[:100]}...")  # Print first 100 chars
    print("-" * 50)
```

## Advanced Usage

### GPU Acceleration

By default, the extractor will use CUDA if available. You can explicitly specify the device:

```python
# Use CPU explicitly
extractor = SmolDoclingExtractor(device="cpu")

# Use specific GPU
extractor = SmolDoclingExtractor(device="cuda:0")
```

### Custom Model Configuration

You can specify a different model from the same family:

```python
# Use a different model variant
extractor = SmolDoclingExtractor(model_name="ds4sd/SmolDocling-512M")
```

### Adjusting Generated Text Length

For longer documents, you may want to increase the maximum generated text length:

```python
# Extract with a longer maximum length for complex documents
text = extractor.extract_text("complex_document.pdf", max_length=1024)
```

## Performance Considerations

- Processing time depends on the image size, complexity, and hardware
- GPU acceleration is recommended for batch processing
- First initialization loads the model which may take some time
- Subsequent calls are much faster as the model remains in memory

## Comparison with Traditional OCR

`docling_ocr` differs from traditional OCR engines in several key ways:

| Feature | Traditional OCR | docling_ocr |
|---------|----------------|-------------|
| Text Recognition | Character/word based | Context-aware language understanding |
| Layout Understanding | Limited/separate process | Integrated understanding |
| Language Understanding | Limited | Leverages LLM language capabilities |
| Format Flexibility | Engine-specific | Adaptable to various formats |
| Context Retention | Limited | Maintains document context |

## Examples

### Forms and Structured Documents

```python
from docling_ocr import SmolDoclingExtractor

extractor = SmolDoclingExtractor()
form_text = extractor("tax_form.jpg")
print(form_text)
```

### Tables and Spreadsheets

```python
spreadsheet_text = extractor("financial_data.jpg")
print(spreadsheet_text)
```

### Receipts and Invoices

```python
receipt_text = extractor("receipt.jpg")
print(receipt_text)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Roadmap

- Support for PDF documents with multi-page handling
- Additional LLM-based extraction models
- Fine-tuning options for specific document types
- Structured data extraction (JSON output)
- Layout-preserving extraction options

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on the amazing work of the SmolDocling team for the [SmolDocling-256M-preview model.](https://huggingface.co/ds4sd/SmolDocling-256M-preview)
- Inspired by the growing field of document AI
- Thanks to the HuggingFace team for making transformers accessible

## Citation

If you use this package in your research, please cite:

```
@software{docling_ocr,
  author = {Adhing'a Fredrick},
  title = {docling_ocr: LLM-based Document Text Extraction},
  year = {2025},
  url = {https://github.com/FREDERICO23/docling_ocr}
}
```

## Contact

For questions and support, please open an issue on the GitHub repository or contact [adhingafredrick@gmail.com](mailto:adhingafredrick@gmail.com).
