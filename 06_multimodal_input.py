"""
================================================================================
Module 06: Multimodal Input & Visual Reasoning (Google Gemini Edition)
================================================================================

CONCEPT OVERVIEW:
-----------------
Unlike legacy LLMs that only process text tokens, modern foundation models like
Google Gemini are "natively multimodal". Gemini was trained from the ground up
on interleaved text, images, video, and audio.

In this module, you will learn how to:
1. Pass local images (via base64 encoding) or remote images (via URLs) to Gemini.
2. Structure a multimodal `HumanMessage` containing both image data and a text prompt.
3. Perform analytical reasoning over visual content (e.g. charts, diagrams, photos).

WHY THIS PATTERN MATTERS:
-------------------------
Multimodal capability turns your AI application into a complete perceptual system.
Instead of relying on brittle OCR (Optical Character Recognition) pipelines or
separate vision models, a single call to Gemini can extract data, analyze visual
trends, read handwritten notes, or inspect UI wireframes.

GEMINI SPECIFICS:
-----------------
Google Gemini models (such as `gemini-3.5-flash-lite` and `gemini-3.6-flash`) have
exceptional image comprehension. LangChain supports both:
- Standard Data URI format: `{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}`
- Structured image block format: `{"type": "image", "base64": "...", "mime_type": "image/jpeg"}`
================================================================================
"""

import base64
import mimetypes
import os
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 1. Environment & Setup
# ------------------------------------------------------------------------------
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

# Use the vision-capable model (gemini-3.5-flash-lite or gemini-3.6-flash)
MODEL_NAME = os.getenv("VISION_MODEL", os.getenv("GEMINI_MODEL", "gemini-3.6-flash"))
print(f"[Vision Setup] Initializing multimodal Gemini model: '{MODEL_NAME}'...\n")

model = init_chat_model(MODEL_NAME, model_provider="google_genai")


# ------------------------------------------------------------------------------
# 2. Image Loading & Encoding Utilities
# ------------------------------------------------------------------------------
def encode_local_image(image_path: Path) -> dict:
    """Read a local image file and format it as a base64 LangChain image block."""
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found at {image_path}")

    # Determine MIME type (e.g., 'image/jpeg' or 'image/png')
    mime_type, _ = mimetypes.guess_type(str(image_path))
    if not mime_type:
        mime_type = "image/jpeg"

    raw_bytes = image_path.read_bytes()
    b64_encoded = base64.b64encode(raw_bytes).decode("utf-8")

    # Format as a standard multimodal image dictionary
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{b64_encoded}"
        },
    }


def format_url_image(url: str) -> dict:
    """Format a publicly reachable image URL for model ingestion."""
    return {
        "type": "image_url",
        "image_url": {
            "url": url
        },
    }


# ------------------------------------------------------------------------------
# 3. Invoking the Model with Multimodal Payload
# ------------------------------------------------------------------------------
def main():
    # Resolve the sample image path in the assets folder
    project_root = Path(__file__).resolve().parent
    sample_image_path = project_root / "assets" / "sample_chart.jpg"

    print(f"Loading local sample image: {sample_image_path}")
    image_block = encode_local_image(sample_image_path)

    # A multimodal HumanMessage combines a text prompt and one or more image blocks
    user_prompt = (
        "Please analyze this quarterly performance chart:\n"
        "1. Identify the quarters shown and their respective throughput values.\n"
        "2. Which quarter showed the highest throughput, and what is the overall growth trend?"
    )

    print(f"\n--- Multimodal Prompt ---\n{user_prompt}\n")

    message = HumanMessage(
        content=[
            {"type": "text", "text": user_prompt},
            image_block,
        ]
    )

    print("--- Sending Multimodal Request to Google Gemini ---")
    response = model.invoke([message])

    answer = getattr(response, "text", response.content)
    print("\n--- Gemini Visual Analysis ---")
    print(answer)


if __name__ == "__main__":
    main()
