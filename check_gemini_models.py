# check_gemini_models.py
import os

import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise SystemExit(
        "No API key found. Set GOOGLE_API_KEY (preferred) or GEMINI_API_KEY, e.g.\n"
        "  export GOOGLE_API_KEY='AIza...'\n"
    )

genai.configure(api_key=api_key)

print("✅ Listing Gemini models available to your API key:\n")
for m in genai.list_models():
    print(m.name)
