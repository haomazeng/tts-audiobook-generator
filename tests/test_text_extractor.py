import pytest
from src.text_extractor import extract_text_from_pdf, extract_text_from_md, split_into_sentences

def test_extract_text_from_md():
    result = extract_text_from_md("tests/fixtures/sample.md")
    assert isinstance(result, str)
    assert len(result) > 0
    assert "测试文档" in result

@pytest.mark.skip(reason="No PDF fixture file")
def test_extract_text_from_pdf():
    result = extract_text_from_pdf("tests/fixtures/sample.pdf")
    assert isinstance(result, str)
    assert len(result) > 0

def test_split_into_sentences():
    text = "这是第一句。这是第二句！这是第三句？"
    result = split_into_sentences(text)
    # All sentences fit in one chunk (max_chunk_size=500)
    assert len(result) == 1
    assert "这是第一句。" in result[0]
    assert "这是第二句！" in result[0]

def test_split_respects_chunk_size():
    text = "句子1。" * 100
    result = split_into_sentences(text, max_chunk_size=200)
    for chunk in result:
        assert len(chunk) <= 200
