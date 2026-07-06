from pipeline.cost import estimate_image_cost, estimate_llm_cost, format_estimate


def test_image_cost_gemini_lite():
    cost = estimate_image_cost(10, "openrouter_gemini_lite")
    assert cost == 0.35


def test_image_cost_gemini_flash():
    cost = estimate_image_cost(10, "openrouter_gemini_flash")
    assert cost == 0.7


def test_image_cost_gemini_pro():
    cost = estimate_image_cost(10, "openrouter_gemini_pro")
    assert cost == 1.4


def test_image_cost_unknown():
    try:
        estimate_image_cost(1, "fake")
        assert False
    except ValueError:
        pass


def test_llm_cost_openrouter():
    cost = estimate_llm_cost(100000)
    assert cost == 0.2


def test_llm_cost_ollama():
    cost = estimate_llm_cost(100000, model="ollama")
    assert cost == 0.0


def test_format_estimate_basic():
    result = format_estimate(1.0)
    assert "Images" in result


def test_format_estimate_with_llm():
    result = format_estimate(1.0, llm=0.5)
    assert "LLM" in result
    assert "$1.50" in result
