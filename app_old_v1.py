from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.openapi.docs import get_swagger_ui_html, get_redoc_html
from fastapi.openapi.utils import get_openapi
import os

import logging
from logging.handlers import RotatingFileHandler
# ---------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, "service.log")

# Create rotating log handler: 5 MB per file, keep last 5 logs.
handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        handler,
        logging.StreamHandler()     # So logs also show in console
    ]
)

logger = logging.getLogger(__name__)


import secrets
import requests
import base64
import json
import re
import os
import io
from PIL import Image, ImageFile
from datetime import datetime
from pydantic import BaseModel, Field

# Allow PIL to load partially truncated images safely
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------

API_URL = "http://10.250.12.5:8105/v1/chat/completions"
API_KEY = "token-abc123"
MODEL = "Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
# MODEL = "Qwen/Qwen3-VL-8B-Instruct-FP8"

os.makedirs("images", exist_ok=True)


# ---------------------------------------------------------
# BASIC AUTH FOR DOCS
# ---------------------------------------------------------

security = HTTPBasic()

def verify_docs_auth(credentials: HTTPBasicCredentials = Depends(security)):
    if not (
        secrets.compare_digest(credentials.username, "admin")
        and secrets.compare_digest(credentials.password, "Admin@2025")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized",
            headers={"WWW-Authenticate": "Basic"},
        )
    return True


# ---------------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------------

app = FastAPI(
    title="HMEL Card Reader Service",
    version="1.0.1",
    docs_url=None,
    redoc_url=None
)


@app.get("/docs", include_in_schema=False)
def custom_swagger_ui(auth: bool = Depends(verify_docs_auth)):
    return get_swagger_ui_html(openapi_url="/openapi.json", title="Swagger UI")


@app.get("/redoc", include_in_schema=False)
def custom_redoc_ui(auth: bool = Depends(verify_docs_auth)):
    return get_redoc_html(openapi_url="/openapi.json", title="ReDoc")


@app.get("/openapi.json", include_in_schema=False)
def custom_openapi(auth: bool = Depends(verify_docs_auth)):
    return get_openapi(title=app.title, version=app.version, routes=app.routes)


# ---------------------------------------------------------
# ROOT & STATUS
# ---------------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "HMEL Card Reader Service",
        "version": "1.0.1",
        "status": "operational",
        "environment": "staging",
        "docs_url": "/docs"
    }


@app.get("/status")
def status():
    return {"status": "API is working", "service": "extractor", "code": 200}


# ---------------------------------------------------------
# BASE64 INPUT SCHEMA
# ---------------------------------------------------------

class ImagePayload(BaseModel):
    image_base64: str = Field(
        ...,
        title="Image Base64 String",
        description="Base64 encoded string only.",
        example="string"
    )


# ---------------------------------------------------------
# CLEAN LLM JSON WRAPPERS
# ---------------------------------------------------------
def clean_json_markdown(text: str) -> str:
    text = text.strip()

    # Remove ``` / ```json fences if present
    if text.startswith("```"):
        lines = text.splitlines()

        # Drop first line if it's ``` or ```json
        if lines and lines[0].startswith("```"):
            lines = lines[1:]

        # Drop last line if it's ```
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]

        text = "\n".join(lines).strip()

    # Try to keep only the JSON object between first { and last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        text = text[start:end + 1]

    return text.strip()

import json
from json import JSONDecodeError

def safe_json_loads(raw_text: str):
    """
    Best-effort JSON loading:
    1. Clean markdown fences.
    2. Sanitize newlines inside strings.
    3. Try full parse.
    4. If that fails, trim from the right until we get a valid prefix.
    """
    text = clean_json_markdown(raw_text)

    # Step 2: Replace raw newlines inside strings with spaces
    fixed_chars = []
    in_string = False
    escape = False

    for ch in text:
        if in_string:
            if escape:
                fixed_chars.append(ch)
                escape = False
            else:
                if ch == '\\':
                    fixed_chars.append(ch)
                    escape = True
                elif ch == '"':
                    fixed_chars.append(ch)
                    in_string = False
                elif ch in ['\n', '\r']:
                    # Avoid raw newlines inside JSON strings
                    fixed_chars.append(' ')
                else:
                    fixed_chars.append(ch)
        else:
            if ch == '"':
                fixed_chars.append(ch)
                in_string = True
            else:
                fixed_chars.append(ch)

    text = "".join(fixed_chars).strip()

    # Step 3: normal parse
    try:
        return json.loads(text)
    except JSONDecodeError:
        pass

    # Step 4: last-resort – walk backwards and try prefixes
    # This lets us at least parse the *valid* front part if the tail is truncated.
    for i in range(len(text), 0, -1):
        candidate = text[:i].rstrip()
        # Cheap heuristic: must end with '}' or ']' to be even worth trying
        if not candidate.endswith(("}", "]")):
            continue

        try:
            return json.loads(candidate)
        except JSONDecodeError:
            continue

    # If nothing worked, raise the original error
    raise JSONDecodeError("Unable to recover valid JSON from LLM output", text, 0)


# ---------------------------------------------------------
# FINAL FIXED /extract ENDPOINT
# ---------------------------------------------------------
@app.post("/extract")
def extract_details(payload: ImagePayload):

    logger.info("📥 /extract endpoint called")
    logger.debug(f"Incoming payload size: {len(payload.image_base64)} characters")

    # -----------------------------------------
    # 1️⃣ Extract clean base64 + mime type
    # -----------------------------------------
    logger.debug("Validating base64 input format")

    # Capture MIME type and base64 data separately
    match = re.match(r"data:(image\/[a-zA-Z0-9.+-]+);base64,(.*)", payload.image_base64)
    if not match:
        logger.error("Invalid base64 format received")
        raise HTTPException(400, "Invalid base64 format.")

    mime_type = match.group(1)          # e.g. "image/png" or "image/jpeg"
    clean_b64 = match.group(2)          # pure base64 string (no header)
    logger.debug(f"Base64 header removed successfully, mime_type={mime_type}")

    # -----------------------------------------
    # 2️⃣ Decode image safely (for saving only)
    # -----------------------------------------
    logger.debug("Decoding base64 image for saving")
    try:
        image_bytes = base64.b64decode(clean_b64)
        img = Image.open(io.BytesIO(image_bytes))
        logger.info("Image decoded successfully")
    except Exception as e:
        logger.error(f"Image decoding failed: {e}")
        raise HTTPException(400, "Invalid base64 image data.")

    # -----------------------------------------
    # 3️⃣ Timestamp-based filename
    # -----------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_hex = secrets.token_hex(4)
    base_name = f"extract_{timestamp}_{random_hex}"
    image_path = os.path.join("images", f"{base_name}.png")
    json_path  = os.path.join("images", f"{base_name}.json")

    logger.debug(f"Generated filenames: {image_path}, {json_path}")

    # -----------------------------------------
    # 4️⃣ Save PNG WITHOUT DPI (for logging/audit only)
    # -----------------------------------------
    logger.debug("Saving PNG to disk")
    try:
        img.save(image_path, format="PNG")
        logger.info(f"Image saved successfully: {image_path}")
    except Exception as e:
        logger.critical(f"Failed to save PNG: {e}")
        raise HTTPException(500, f"Error saving PNG: {e}")

    # ⚠️ IMPORTANT:
    # We STOP using the saved PNG for LLM.
    # We will use the ORIGINAL base64 from the request instead.
    # So we DO NOT reload and DO NOT re-encode the PNG here.

    # -----------------------------------------
    # 7️⃣ Vision Model prompt
    # -----------------------------------------
    logger.debug("Preparing LLM payload for Vision model")

    prompt_text = """
You must return ONLY valid JSON. No explanation, no markdown, no text outside JSON.

Return JSON EXACTLY in this structure:

{
    "extracted_info": {
        "name": [],
        "email": [],
        "phone": [],
        "designation": "",
        "company_name": "",
        "website": "",
        "address": "",
        "additional_info": {
            "category": [],
            "other": []
        }
    }
}

STRICT RULES:
- Return ONLY JSON. NO extra text, NO markdown.
- Do NOT add or rename any fields.
- All names, emails, and phones must be arrays.
- If something is missing, return empty string "" or [].
- DO NOT put newline characters inside any string values.
  - Replace any line breaks with a single space.
  - For example, "Senior Business Manager - \\n Corporate" MUST become "Senior Business Manager - Corporate".
- Every string value must be a single line, no actual line breaks.
"""

    # 👉 Use the ORIGINAL base64 we received (not the saved PNG)
    data_url_for_llm = f"data:{mime_type};base64,{clean_b64}"

    payload_data = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": data_url_for_llm}}
                ]
            }
        ],
        "max_tokens": 4000,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}  # if your server supports it
    }

    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}

    # -----------------------------------------
    # 8️⃣ Send request to VLLM Vision Model
    # -----------------------------------------
    logger.info("Sending request to Vision LLM model")
    response = requests.post(API_URL, headers=headers, json=payload_data)

    if response.status_code != 200:
        logger.error(f"Vision API Error: {response.text}")
        raise HTTPException(500, "LLM API Error: " + response.text)

    logger.debug("Vision model responded successfully")

    resp_json = response.json()
    if "choices" not in resp_json:
        logger.error("Invalid LLM response structure")
        raise HTTPException(500, "Invalid LLM response: " + response.text)

    cleaned_text = clean_json_markdown(resp_json["choices"][0]["message"]["content"])
    logger.info(f"Response of LLM is: {cleaned_text}")

    # -----------------------------------------
    # 9️⃣ Parse JSON
    # -----------------------------------------
    logger.debug("Parsing LLM JSON response")
    try:
        parsed = safe_json_loads(cleaned_text)
        logger.info("LLM JSON parsed successfully")
    except Exception as e:
        logger.error(f"JSON parsing failed: {e}")
        raise HTTPException(
            500,
            {"error": "Failed to parse JSON", "raw_output": cleaned_text}
        )

    logger.info("Extraction completed successfully")

    return parsed["extracted_info"]
