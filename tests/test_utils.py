import unittest
from unittest.mock import patch, MagicMock, mock_open, call
import os
import tempfile
import shutil
from PIL import Image

from docling_ocr.utils import batch_process

class TestUtils(unittest.TestCase):
    """Tests for utility functions in the docling_ocr package."""
    
    def setUp(self):
        """Set up test resources."""
        self.test_dir = tempfile.mkdtemp()
        self.output_dir = tempfile.mkdtemp()
        
        # Create some test images
        self.create_test_image("test1.jpg")
        self.create_test_image("test2.png")
        self.create_test_image("test3.txt")  # Not an image file
        
    def tearDown(self):
        """Clean up test resources."""
        shutil.rmtree(self.test_dir)
        shutil.rmtree(self.output_dir)
        
    def create_test_image(self, filename):
        """Helper to create test image files."""
        filepath = os.path.join(self.test_dir, filename)
        # Just create an empty file for testing
        with open(filepath, 'w') as f:
            f.write("dummy content")
            
    def test_batch_process(self):
        """Test the batch_process function."""
        # Create a mock extractor
        mock_extractor = MagicMock()
        mock_extractor.extract_text.side_effect = [
            "Content from test1.jpg",
            "Content from test2.png",
            Exception("Test error for test3.txt")
        ]
        
        # Call batch_process
        results = batch_process(
            mock_extractor,
            self.test_dir,
            self.output_dir,
            extensions=['.jpg', '.png', '.txt']
        )
        
        # Check results
        self.assertEqual(len(results), 3)
        self.assertEqual(results["test1.jpg"], "Content from test1.jpg")
        self.assertEqual(results["test2.png"], "Content from test2.png")
        self.assertTrue(results["test3.txt"].startswith("ERROR"))
        
        # Check that files were created
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test2.txt")))
        
        # Check file content
        with open(os.path.join(self.output_dir, "test1.txt"), 'r') as f:
            self.assertEqual(f.read(), "Content from test1.jpg")
            
    def test_batch_process_no_output_dir(self):
        """Test batch_process without an output directory."""
        mock_extractor = MagicMock()
        mock_extractor.extract_text.return_value = "Test content"
        
        # We should still get results even without an output directory
        results = batch_process(
            mock_extractor,
            self.test_dir,
            output_dir=None,
            extensions=['.jpg', '.png']
        )
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results["test1.jpg"], "Test content")
        self.assertEqual(results["test2.png"], "Test content")
        
    def test_batch_process_filter_extensions(self):
        """Test that batch_process correctly filters by file extension."""
        mock_extractor = MagicMock()
        
        # Process only JPG files
        results = batch_process(
            mock_extractor,
            self.test_dir,
            extensions=['.jpg']
        )
        
        # Should only process one file
        self.assertEqual(len(results), 1)
        self.assertIn("test1.jpg", results)
        self.assertNotIn("test2.png", results)
        self.assertNotIn("test3.txt", results)
        
    def test_batch_process_create_output_dir(self):
        """Test that batch_process creates the output directory if it doesn't exist."""
        mock_extractor = MagicMock()
        mock_extractor.extract_text.return_value = "Test content"
        
        # Delete the output directory
        shutil.rmtree(self.output_dir)
        self.assertFalse(os.path.exists(self.output_dir))
        
        # Call batch_process
        batch_process(
            mock_extractor,
            self.test_dir,
            self.output_dir,
            extensions=['.jpg']
        )
        
        # The output directory should have been created
        self.assertTrue(os.path.exists(self.output_dir))
        self.assertTrue(os.path.exists(os.path.join(self.output_dir, "test1.txt")))