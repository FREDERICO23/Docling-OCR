from docling_ocr import SmolDoclingExtractor
from docling_ocr.utils import batch_process
import sys
import os

def main():
    # Check command line arguments
    if len(sys.argv) < 3:
        print("Usage: python batch_processing.py <input_dir> <output_dir>")
        return
        
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]
    
    # Check if input directory exists
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return
    
    # Initialize extractor
    print("Initializing SmolDocling extractor...")
    extractor = SmolDoclingExtractor()
    
    # Process images
    print(f"Processing images from {input_dir}...")
    results = batch_process(extractor, input_dir, output_dir)
    
    # Print summary
    print("\nProcessing Summary:")
    print("-" * 50)
    print(f"Total files processed: {len(results)}")
    errors = sum(1 for text in results.values() if text.startswith("ERROR"))
    print(f"Successful extractions: {len(results) - errors}")
    print(f"Failed extractions: {errors}")
    print(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main()