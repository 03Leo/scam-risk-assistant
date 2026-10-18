"""
tests/test_ocr.py
=================
Unit tests for src/ocr.py.

Covers:
- Tesseract executable discovery and diagnostics
- Image loading (bytes, PIL Image, invalid inputs)
- Pillow image preprocessing (color conversion, contrast)
- OCR extraction, confidence estimation, and graceful fallbacks
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch
import pytest
from PIL import Image, ImageDraw

from src.ocr import (
    OcrResult,
    TesseractNotFoundError,
    extract_text_from_image,
    find_tesseract_binary,
    is_tesseract_available,
    load_image,
    preprocess_image,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_sample_image(text: str = "URGENT BANK ALERT") -> Image.Image:
    """Create a high-contrast in-memory image with text."""
    img = Image.new("RGB", (300, 100), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 40), text, fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------------------
# Discovery Tests
# ---------------------------------------------------------------------------

class TestBinaryDiscovery:
    def test_is_tesseract_available_returns_tuple(self):
        avail, msg = is_tesseract_available()
        assert isinstance(avail, bool)
        assert isinstance(msg, str)
        if not avail:
            assert "winget" in msg.lower() or "tesseract" in msg.lower()

    def test_find_tesseract_binary_precedence(self):
        with patch("shutil.which", return_value="C:\\fake\\tesseract.exe"):
            found = find_tesseract_binary()
            assert found == "C:\\fake\\tesseract.exe"


# ---------------------------------------------------------------------------
# Image Loading & Preprocessing Tests
# ---------------------------------------------------------------------------

class TestImageLoading:
    def test_loads_from_pil_instance(self):
        img = _create_sample_image()
        loaded = load_image(img)
        assert loaded == img

    def test_loads_from_raw_bytes(self):
        img = _create_sample_image()
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        loaded = load_image(buf.getvalue())
        assert isinstance(loaded, Image.Image)
        assert loaded.size == img.size

    def test_raises_type_error_for_invalid_input(self):
        with pytest.raises(TypeError, match="Unsupported image input type"):
            load_image(12345)


class TestImagePreprocessing:
    def test_converts_rgba_to_grayscale(self):
        rgba_img = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
        processed = preprocess_image(rgba_img)
        assert processed.mode == "L"
        assert processed.size == (100, 100)

    def test_preserves_dimensions(self):
        rgb_img = Image.new("RGB", (250, 120), (200, 200, 200))
        processed = preprocess_image(rgb_img)
        assert processed.size == (250, 120)


# ---------------------------------------------------------------------------
# OCR Extraction Tests
# ---------------------------------------------------------------------------

class TestOcrExtraction:
    def test_raises_error_when_binary_missing(self):
        with patch("src.ocr.find_tesseract_binary", return_value=None):
            img = _create_sample_image()
            with pytest.raises(TesseractNotFoundError, match="Tesseract OCR executable not found"):
                extract_text_from_image(img)

    def test_mocked_successful_extraction(self):
        mock_data = {
            "text": ["", "URGENT", "ACTION", "REQUIRED", ""],
            "conf": ["-1", "95", "92", "88", "-1"],
        }
        with patch("src.ocr.find_tesseract_binary", return_value="C:\\mock\\tesseract.exe"), \
             patch("pytesseract.image_to_data", return_value=mock_data):
            img = _create_sample_image()
            res = extract_text_from_image(img)

            assert isinstance(res, OcrResult)
            assert res.extracted_text == "URGENT ACTION REQUIRED"
            assert res.word_count == 3
            assert res.confidence == pytest.approx(91.7, abs=0.1)
            assert res.is_empty is False
            assert res.is_low_quality is False
            assert res.warning is None

    def test_mocked_empty_extraction(self):
        mock_data = {
            "text": ["", "  ", ""],
            "conf": ["-1", "-1", "-1"],
        }
        with patch("src.ocr.find_tesseract_binary", return_value="C:\\mock\\tesseract.exe"), \
             patch("pytesseract.image_to_data", return_value=mock_data):
            img = Image.new("RGB", (100, 100), (255, 255, 255))
            res = extract_text_from_image(img)

            assert res.is_empty is True
            assert res.is_low_quality is True
            assert "No readable text detected" in res.warning

    def test_mocked_low_confidence_extraction(self):
        mock_data = {
            "text": ["blurry", "word"],
            "conf": ["40", "45"],
        }
        with patch("src.ocr.find_tesseract_binary", return_value="C:\\mock\\tesseract.exe"), \
             patch("pytesseract.image_to_data", return_value=mock_data):
            img = _create_sample_image()
            res = extract_text_from_image(img, confidence_threshold=60.0)

            assert res.is_low_quality is True
            assert "Low OCR confidence" in res.warning
