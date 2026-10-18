"""
src/ocr.py
==========
Optical Character Recognition (OCR) module for scam message screenshots.

Phase 2 Stack:
- Pillow (PIL) for image ingestion, resizing, and contrast enhancement.
- pytesseract as the local OCR extraction bridge.

Strict Separation of Concerns:
- All image loading, preprocessing, Tesseract executable resolution,
  and OCR confidence estimation live here.
- app.py strictly consumes extract_text_from_image() and passes the resulting
  text to src.analyzer.analyze_message().

Design Principles:
1. Local Privacy: All image reading and OCR processing run 100% locally.
   No image pixels leave the device.
2. Robust Binary Discovery: Automatically checks PATH, environment variables,
   and standard Windows installation paths for tesseract.exe.
3. Quality & Uncertainty Awareness: Evaluates OCR confidence scores across
   detected words to flag blurry or low-resolution screenshots.
"""

from __future__ import annotations

import io
import logging
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Union

from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import pytesseract

log = logging.getLogger("ocr")

# ---------------------------------------------------------------------------
# Standard Windows paths where Tesseract is commonly installed
# ---------------------------------------------------------------------------

COMMON_TESSERACT_PATHS: list[Path] = [
    Path(os.environ.get("PROGRAMFILES", "C:\\Program Files")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", "C:\\Program Files (x86)")) / "Tesseract-OCR" / "tesseract.exe",
    Path(os.environ.get("LOCALAPPDATA", "C:\\Users\\Default\\AppData\\Local")) / "Programs" / "Tesseract-OCR" / "tesseract.exe",
]


class TesseractNotFoundError(FileNotFoundError):
    """Raised when the Tesseract OCR executable cannot be located."""
    pass


@dataclass
class OcrResult:
    extracted_text: str
    confidence: float          # Average token confidence (0.0 to 100.0)
    is_empty: bool             # True if no text was found
    word_count: int            # Number of detected words
    is_low_quality: bool       # True if confidence is low (< 60.0) or empty
    warning: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# 1. Executable Discovery
# ---------------------------------------------------------------------------

def find_tesseract_binary() -> Optional[str]:
    """
    Search for the Tesseract OCR executable.

    Search precedence:
    1. pytesseract.pytesseract.tesseract_cmd (if already configured and exists)
    2. Environment variable TESSERACT_CMD or TESSERACT_PATH
    3. System PATH (via shutil.which)
    4. Common Windows installation directories

    Returns
    -------
    Absolute path string to tesseract executable, or None if not found.
    """
    # 1. Check existing setting if it points to a valid file
    current_cmd = getattr(pytesseract.pytesseract, "tesseract_cmd", "tesseract")
    if current_cmd and current_cmd != "tesseract" and os.path.isfile(current_cmd):
        return current_cmd

    # 2. Check environment variables
    env_path = os.environ.get("TESSERACT_CMD") or os.environ.get("TESSERACT_PATH")
    if env_path and os.path.isfile(env_path):
        pytesseract.pytesseract.tesseract_cmd = env_path
        return env_path

    # 3. Check system PATH
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return which_path

    # 4. Check common Windows paths
    for cand in COMMON_TESSERACT_PATHS:
        if cand.is_file():
            found = str(cand.resolve())
            pytesseract.pytesseract.tesseract_cmd = found
            log.info("Discovered Tesseract binary at '%s'.", found)
            return found

    return None


def is_tesseract_available() -> tuple[bool, str]:
    """
    Check if Tesseract is available for OCR.

    Returns
    -------
    (available: bool, path_or_guidance: str)
    """
    binary = find_tesseract_binary()
    if binary:
        return True, binary
    return False, (
        "Tesseract OCR executable not found on this system.\n"
        "To enable screenshot OCR:\n"
        "1. Install via winget: winget install UB-Mannheim.TesseractOCR\n"
        "   or download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
        "2. Alternatively set environment variable TESSERACT_CMD to tesseract.exe path."
    )


# ---------------------------------------------------------------------------
# 2. Image Preprocessing (Pillow)
# ---------------------------------------------------------------------------

def preprocess_image(image: Image.Image) -> Image.Image:
    """
    Apply conservative enhancements to improve OCR accuracy on screenshots.

    Operations:
    1. Convert to RGB then grayscale (L).
    2. Auto-contrast normalization to maximize text/background separation.
    3. Slight sharpness enhancement to crisp up letter boundaries.
    """
    # Coerce to RGB first if RGBA or palette image
    if image.mode in ("RGBA", "P", "LA"):
        image = image.convert("RGB")

    gray = image.convert("L")

    # Normalize contrast
    enhanced = ImageOps.autocontrast(gray, cutoff=1)

    # Enhance sharpness slightly
    enhancer = ImageEnhance.Sharpness(enhanced)
    sharp = enhancer.enhance(1.5)

    return sharp


def load_image(
    image_input: Union[bytes, BinaryIO, Image.Image, Path, str]
) -> Image.Image:
    """
    Coerce various image representations into a PIL Image.
    """
    if isinstance(image_input, Image.Image):
        return image_input

    if isinstance(image_input, bytes):
        return Image.open(io.BytesIO(image_input))

    if isinstance(image_input, (str, Path)):
        return Image.open(Path(image_input))

    if hasattr(image_input, "read"):
        return Image.open(image_input)

    raise TypeError(f"Unsupported image input type: {type(image_input)}")


# ---------------------------------------------------------------------------
# 3. OCR Text Extraction & Quality Scoring
# ---------------------------------------------------------------------------

def extract_text_from_image(
    image_input: Union[bytes, BinaryIO, Image.Image, Path, str],
    preprocess: bool = True,
    confidence_threshold: float = 60.0,
) -> OcrResult:
    """
    Extract text and compute quality metrics from an image.

    Parameters
    ----------
    image_input          : Raw bytes, file object, path, or PIL Image.
    preprocess           : Whether to apply contrast/sharpness enhancements.
    confidence_threshold : Minimum average word confidence (0-100) before
                           flagging the extraction as low quality.

    Returns
    -------
    OcrResult with extracted_text, confidence score, word count, and warnings.

    Raises
    ------
    TesseractNotFoundError
        If the Tesseract binary is not installed or discoverable.
    """
    available, binary_or_msg = is_tesseract_available()
    if not available:
        raise TesseractNotFoundError(binary_or_msg)

    raw_image = load_image(image_input)
    processed_image = preprocess_image(raw_image) if preprocess else raw_image

    # Use image_to_data to obtain per-word confidence scores
    try:
        data = pytesseract.image_to_data(
            processed_image,
            lang="eng",
            output_type=pytesseract.Output.DICT,
        )
    except Exception as exc:
        log.error("Tesseract execution failed: %s", exc)
        raise RuntimeError(f"OCR extraction failed: {exc}") from exc

    words: list[str] = []
    confidences: list[float] = []

    for text, conf in zip(data["text"], data["conf"]):
        text_clean = text.strip()
        conf_float = float(conf)
        # Tesseract flags non-word layout blocks with -1 confidence
        if text_clean and conf_float >= 0:
            words.append(text_clean)
            confidences.append(conf_float)

    full_text = " ".join(words).strip()
    full_text = re.sub(r"\s+", " ", full_text)

    is_empty = len(words) == 0
    mean_conf = round(float(sum(confidences) / len(confidences)), 1) if confidences else 0.0
    is_low_quality = is_empty or (mean_conf < confidence_threshold)

    warning: Optional[str] = None
    if is_empty:
        warning = "No readable text detected in the provided image."
    elif is_low_quality:
        warning = (
            f"Low OCR confidence ({mean_conf:.1f}%). Text may contain transcription errors "
            "due to blur, small fonts, or low contrast."
        )

    return OcrResult(
        extracted_text=full_text,
        confidence=mean_conf,
        is_empty=is_empty,
        word_count=len(words),
        is_low_quality=is_low_quality,
        warning=warning,
    )
