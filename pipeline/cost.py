PRICES = {
    "openrouter_gemini_lite": 0.035,
    "openrouter_gemini_flash": 0.07,
    "openrouter_gemini_pro": 0.14,
    "openrouter_grok_imagine": 0.05,
    "openrouter_llm": 0.000002,
    "deepseek_llm": 0.00000014,
    "ollama_llm": 0,
    "kenari_llm": 0,
}

# Kenari bills per image in Rupiah. Source: GET https://kenari.id/v1/models
# (pricing_lines[0].micro_idr for models whose endpoints include "images").
KENARI_IMAGE_IDR = {
    "gpt-image-2": 125,
    "nano-banana-2-lite": 150,
    "nano-banana-2": 250,
    "nano-banana-pro": 350,
    "grok-imagine-image": 300,
    "grok-imagine-image-quality": 750,
    "grok-imagine-image-2-0": 900,
}
KENARI_IMAGE_IDR_DEFAULT = 250
IDR_PER_USD = 16000  # rough; only used to show an approximate USD figure
