from docling_ocr import SmolDoclingExtractor
import sys

def main():
    # Check command line arguments
    if len(sys.argv) < 2:
        print("Usage: python simple_extraction.py <image_path>")
        return
        
    image_path = sys.argv[1]
    
    # Initialize extractor
    print("Initializing SmolDocling extractor...")
    extractor = SmolDoclingExtractor()
    
    # Extract text
    print(f"Extracting text from {image_path}...")
    text = extractor.extract_text(image_path)
    
    print("\nExtracted Text:")
    print("-" * 50)
    print(text)
    print("-" * 50)

if __name__ == "__main__":
    main()
