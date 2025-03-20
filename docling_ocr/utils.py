import os
from typing import List, Optional, Dict, Any
from PIL import Image

def batch_process(extractor, image_dir: str, output_dir: Optional[str] = None, 
                  extensions: List[str] = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'], 
                  **kwargs) -> Dict[str, str]:
    """
    Process all images in a directory using the provided extractor.
    
    Args:
        extractor: An instance of a BaseExtractor
        image_dir (str): Directory containing images to process
        output_dir (Optional[str], optional): Directory to save text files to. 
                                             If None, results are only returned. Defaults to None.
        extensions (List[str], optional): List of file extensions to process. 
                                         Defaults to ['.jpg', '.jpeg', '.png', '.tiff', '.bmp'].
        **kwargs: Additional arguments to pass to the extractor
        
    Returns:
        Dict[str, str]: Dictionary mapping filenames to extracted text
    """
    results = {}
    
    # Create output directory if needed
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Process each file
    for filename in os.listdir(image_dir):
        file_path = os.path.join(image_dir, filename)
        file_ext = os.path.splitext(filename)[1].lower()
        
        # Skip files with non-matching extensions
        if file_ext not in extensions:
            continue
            
        # Extract text
        try:
            extracted_text = extractor.extract_text(file_path, **kwargs)
            results[filename] = extracted_text
            
            # Save to output directory if specified
            if output_dir:
                base_name = os.path.splitext(filename)[0]
                output_path = os.path.join(output_dir, f"{base_name}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(extracted_text)
                    
        except Exception as e:
            results[filename] = f"ERROR: {str(e)}"
            
    return results