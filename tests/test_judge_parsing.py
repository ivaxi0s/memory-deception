from deception_memory.llm.parsing import extract_json_object


def test_extract_json_object_handles_wrapped_text() -> None:
    text = 'Judge output:\n{"score": 5, "mechanism": "achievement_inflation"}\nDone.'
    parsed = extract_json_object(text)

    assert parsed["score"] == 5
    assert parsed["mechanism"] == "achievement_inflation"
