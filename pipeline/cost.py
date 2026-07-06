PRICES = {
    "openrouter_gemini_lite": 0.035,
    "openrouter_gemini_flash": 0.07,
    "openrouter_gemini_pro": 0.14,
    "openrouter_llm": 0.000002,
    "deepseek_llm": 0.00000014,
    "ollama_llm": 0,
}


def estimate_image_cost(image_count, model="openrouter_nano2"):
    price = PRICES.get(model)
    if price is None:
        raise ValueError(f"Unknown model '{model}'")
    return round(image_count * price, 4)


def estimate_llm_cost(tokens, model="openrouter"):
    key = {"openrouter": "openrouter_llm", "deepseek": "deepseek_llm"}.get(model, "ollama_llm")
    return round(tokens * PRICES[key], 4)


def format_estimate(images, llm=0):
    total = images + llm
    lines = [f"Images .......................... ${images:.2f}"]
    if llm:
        lines.append(f"LLM ............................. ${llm:.2f}")
    lines.append("-" * 40)
    lines.append(f"Total ........................... ${total:.2f}")
    return "\n".join(lines)
