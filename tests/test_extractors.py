import unittest
from unittest.mock import patch, MagicMock, ANY
import os
import torch
from PIL import Image
import io

from docling_ocr.extractors import BaseExtractor, SmolDoclingExtractor

class TestBaseExtractor(unittest.TestCase):
    """Tests for the BaseExtractor abstract base class."""
    
    def test_abstract_methods(self):
        """Test that BaseExtractor cannot be instantiated directly."""
        with self.assertRaises(TypeError):
            BaseExtractor()
            
    def test_subclass_implementation(self):
        """Test that properly implemented subclasses can be instantiated."""
        class TestExtractor(BaseExtractor):
            def extract_text(self, image_path, **kwargs):
                return "Test text"
                
            def extract_text_from_image(self, image, **kwargs):
                return "Test text from image"
                
        extractor = TestExtractor()
        self.assertEqual(extractor.extract_text("dummy_path"), "Test text")
        self.assertEqual(extractor.extract_text_from_image(MagicMock()), "Test text from image")

class TestSmolDoclingExtractor(unittest.TestCase):
    """Tests for the SmolDoclingExtractor class."""
    
    @patch('docling_ocr.extractors.AutoProcessor')
    @patch('docling_ocr.extractors.AutoModelForImageTextToText')
    def setUp(self, mock_model_class, mock_processor_class):
        """Set up a SmolDoclingExtractor with mocked dependencies."""
        # Mock the processor
        self.mock_processor = MagicMock()
        mock_processor_class.from_pretrained.return_value = self.mock_processor
        
        # Mock the model
        self.mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = self.mock_model
        self.mock_model.to.return_value = self.mock_model
        
        # Configure the mocks for generate and batch_decode
        generated_ids = torch.tensor([[1, 2, 3]])
        self.mock_model.generate.return_value = generated_ids
        self.mock_processor.batch_decode.return_value = ["Sample extracted text"]
        
        # Create the extractor
        self.extractor = SmolDoclingExtractor(device="cpu")
    
    def test_initialization(self):
        """Test that the extractor initializes correctly."""
        self.assertEqual(self.extractor.model_name, "ds4sd/SmolDocling-256M-preview")
        self.assertEqual(self.extractor.device, "cpu")
        self.mock_model.eval.assert_called_once()
    
    @patch('docling_ocr.extractors.Image')
    def test_extract_text(self, mock_image):
        """Test the extract_text method."""
        # Setup mock image
        mock_img = MagicMock()
        mock_image.open.return_value = mock_img
        
        # Configure the processor inputs
        self.mock_processor.return_value = {
            "pixel_values": torch.tensor([[[0.0]]]),
            "attention_mask": torch.tensor([[1]])
        }
        
        # Call the method
        result = self.extractor.extract_text("dummy_image.jpg")
        
        # Assertions
        mock_image.open.assert_called_once_with("dummy_image.jpg")
        self.mock_processor.assert_called_once_with(images=mock_img, return_tensors="pt")
        self.mock_model.generate.assert_called_once()
        self.mock_processor.batch_decode.assert_called_once()
        self.assertEqual(result, "Sample extracted text")
    
    def test_extract_text_from_image(self):
        """Test the extract_text_from_image method."""
        # Create a dummy PIL image
        mock_img = MagicMock(spec=Image.Image)
        
        # Configure the processor inputs
        self.mock_processor.return_value = {
            "pixel_values": torch.tensor([[[0.0]]]),
            "attention_mask": torch.tensor([[1]])
        }
        
        # Call the method
        result = self.extractor.extract_text_from_image(mock_img)
        
        # Assertions
        self.mock_processor.assert_called_once_with(images=mock_img, return_tensors="pt")
        self.mock_model.generate.assert_called_once()
        self.mock_processor.batch_decode.assert_called_once()
        self.assertEqual(result, "Sample extracted text")
    
    def test_call_with_string(self):
        """Test the __call__ method with a string path."""
        with patch.object(self.extractor, 'extract_text') as mock_extract:
            mock_extract.return_value = "Called with string"
            result = self.extractor("dummy_path.jpg")
            mock_extract.assert_called_once_with("dummy_path.jpg", max_length=512)
            self.assertEqual(result, "Called with string")
    
    def test_call_with_image(self):
        """Test the __call__ method with a PIL Image."""
        with patch.object(self.extractor, 'extract_text_from_image') as mock_extract:
            mock_extract.return_value = "Called with image"
            mock_img = MagicMock(spec=Image.Image)
            result = self.extractor(mock_img)
            mock_extract.assert_called_once_with(mock_img, max_length=512)
            self.assertEqual(result, "Called with image")
    
    def test_call_with_invalid_type(self):
        """Test the __call__ method with an invalid type."""
        with self.assertRaises(TypeError):
            self.extractor(123)  # Not a string or PIL Image
    
    def test_extraction_error(self):
        """Test error handling in extract_text."""
        with patch('docling_ocr.extractors.Image.open', side_effect=Exception("Test error")):
            with self.assertRaises(ValueError) as context:
                self.extractor.extract_text("nonexistent_image.jpg")
            self.assertIn("Error processing image", str(context.exception))

    @patch('docling_ocr.extractors.torch.cuda.is_available')
    def test_auto_device_selection(self, mock_cuda_available):
        """Test that the device is automatically selected based on CUDA availability."""
        # Test when CUDA is available
        mock_cuda_available.return_value = True
        with patch('docling_ocr.extractors.AutoProcessor'), \
             patch('docling_ocr.extractors.AutoModelForImageTextToText'):
            extractor = SmolDoclingExtractor()
            self.assertEqual(extractor.device, "cuda")
        
        # Test when CUDA is not available
        mock_cuda_available.return_value = False
        with patch('docling_ocr.extractors.AutoProcessor'), \
             patch('docling_ocr.extractors.AutoModelForImageTextToText'):
            extractor = SmolDoclingExtractor()
            self.assertEqual(extractor.device, "cpu")



