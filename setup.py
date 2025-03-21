from setuptools import setup, find_packages

setup(
    name="docling_ocr",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.10.0",
        "transformers>=4.15.0",
        "Pillow>=8.0.0",
    ],
    author="Adhing'a Fredrick",
    author_email="adhingafredrick@gmail.com",
    description="An OCR package using LLM models for text extraction from images",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/FREDERICO23/docling_ocr",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)