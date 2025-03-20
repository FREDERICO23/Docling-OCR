from abc import ABC, abstractmethod
from PIL import Image
import torch
from typing import Optional, List, Union, Dict, Any

class BaseExtractor(ABC):
    """
    Abstract base class for all text extraction models.
    
    This class defines the interface that all text extractors must implement.
    """
    
    @abstractmethod
    def extract_text(self, image_path: str, **kwargs) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path (str): Path to the image file
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: Extracted text from the image
        """
        pass
    
    @abstractmethod
    def extract_text_from_image(self, image: Image.Image, **kwargs) -> str:
        """
        Extract text from a PIL Image object.
        
        Args:
            image (PIL.Image.Image): PIL Image object
            **kwargs: Additional model-specific parameters
            
        Returns:
            str: Extracted text from the image
        """
        pass

class SmolDoclingExtractor(BaseExtractor):
    """
    Text extractor using the SmolDocling model from transformers.
    
    This extractor is designed for document understanding tasks and works well
    with scanned documents, forms, and other text-heavy images.
    
    Attributes:
        model_name (str): The name of the model to load from HuggingFace
        processor: The processor for the model
        model: The loaded model
        device (str): The device to run inference on ('cpu' or 'cuda')
    """
    
    def __init__(self, model_name: str = "ds4sd/SmolDocling-256M-preview", device: Optional[str] = None):
        """
        Initialize the SmolDocling extractor.
        
        Args:
            model_name (str, optional): The name of the model to load from HuggingFace.
                                       Defaults to "ds4sd/SmolDocling-256M-preview".
            device (str, optional): Device to use for inference ('cpu' or 'cuda').
                                   If None, will use CUDA if available.
        """
        try:
            from transformers import AutoProcessor, AutoModelForImageTextToText
        except ImportError:
            raise ImportError(
                "Could not import transformers. Please install it with: pip install transformers"
            )
        
        self.model_name = model_name
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Load model and processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForImageTextToText.from_pretrained(model_name).to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
    
    def extract_text(self, image_path: str, max_length: int = 512) -> str:
        """
        Extract text from an image file.
        
        Args:
            image_path (str): Path to the image file
            max_length (int, optional): Maximum length of generated text. Defaults to 512.
            
        Returns:
            str: Extracted text from the image
        """
        # Load image
        try:
            image = Image.open(image_path)
            return self.extract_text_from_image(image, max_length=max_length)
        except Exception as e:
            raise ValueError(f"Error processing image at {image_path}: {str(e)}")
    
    def extract_text_from_image(self, image: Image.Image, max_length: int = 512) -> str:
        """
        Extract text from a PIL Image object.
        
        Args:
            image (PIL.Image.Image): PIL Image object
            max_length (int, optional): Maximum length of generated text. Defaults to 512.
            
        Returns:
            str: Extracted text from the image
        """
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Move inputs to the correct device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Generate text
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"],
                max_length=max_length,
                attention_mask=inputs.get("attention_mask", None)
            )
        
        # Decode the generated ids to text
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return generated_text
    
    def __call__(self, image_input: Union[str, Image.Image], max_length: int = 512) -> str:
        """
        Convenience method to call the extractor directly.
        
        Args:
            image_input (Union[str, Image.Image]): Either a path to an image file or a PIL Image
            max_length (int, optional): Maximum length of generated text. Defaults to 512.
            
        Returns:
            str: Extracted text from the image
        """
        if isinstance(image_input, str):
            return self.extract_text(image_input, max_length=max_length)
        elif isinstance(image_input, Image.Image):
            return self.extract_text_from_image(image_input, max_length=max_length)
        else:
            raise TypeError("image_input must be either a string path or a PIL Image")


