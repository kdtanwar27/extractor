import os
from google import genai
from google.genai import types

# Initialize Gemini client once (not every request)
client = genai.Client(
    api_key="AIzaSyBgHs1z2R6JazBha93tyXOJXLYskXOmrrs",  # safer than hardcoding
)

MODEL_NAME = "gemini-2.5-flash"

SYSTEM_PROMPT = """
You are an AI assistant for Kochi Metro Rail Limited (KMRL).  
Your task is to process input documents (text extracted from PDF, Word, scanned OCR, etc.) and:  
1. the document category is one of the following:
- Safety Circular
- HR Policy
- Invoice
- Incident Report
- Engineering Drawing Summary 
                                 
The input starts with the document category on the first line,
then the document text follows after a newline.

2. Extract structured key fields depending on the category.  
   Be precise, extract only explicit information from the text. If a field is missing, return null.  

3. Always provide a list of **keywords** (3–7 words) that summarize the core content.  

4. Output must be strictly in JSON format with this schema:

{
  \"category\": \"<one of the five categories>\",
  \"fields\": {
    \"document_id\": \"<any reference number, ID, memo no, or file code>\",
    \"deadline\": \"<date or time limit if present, else null>\",
    \"responsible_party\": \"<department, person, or role responsible>\",
    \"key_entities\": [\"list of organizations, systems, or persons involved\"],
    \"financial_info\": \"<if invoice: amount, currency, PO number>\",
    \"technical_info\": \"<if engineering drawing summary: system names, drawing refs, component IDs>\",
    \"safety_action\": \"<if safety circular or incident report: action items, safety instructions>\",
    \"policy_area\": \"<if HR policy: recruitment, leave, grievance, training, etc.>\"
  },
  \"keywords\": [\"keyword1\", \"keyword2\", \"keyword3\", ...],
}

Guidelines:
- For dates, normalize to YYYY-MM-DD if possible.  
- For IDs, capture any alphanumeric strings that look like document numbers (e.g., CIR-2025-09, INV-4567).  
- For amounts, include numeric value and currency (e.g., \"₹ 50,000\").  
- Extract safety-related instructions as exact phrases if present (e.g., “Ensure fire extinguishers are refilled before 15 Sept 2025”).  
- Always include keywords that would help someone later search for this document.  
- If content is bilingual (English + Malayalam), process both and unify results.  

Return only the JSON output, no explanations.
"""

def extract_entities(category: str, text_input: str) -> str:
    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=category + "\n\n" + text_input),
            ],
        ),
    ]

    generate_content_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_budget=-1,
        ),
        system_instruction=[
            types.Part.from_text(text=SYSTEM_PROMPT),
        ],
    )

    output_text = ""
    for chunk in client.models.generate_content_stream(
        model=MODEL_NAME,
        contents=contents,
        config=generate_content_config,
    ):
        if chunk.text:
            output_text += chunk.text.strip()

    return output_text
