import pytest

pytest.importorskip("contractions")
from src.data_preprocess import clean_text  # noqa: E402


@pytest.mark.parametrize("raw, expected", [
    ("Check https://x.com/abc NOW", "check now"),
    ("@support why is this #broken", "why is this"),
    ("I can't believe it", "i cannot believe it"),
    ("Price is $99!!! :)", "price is"),
    ("  lots   of    spaces ", "lots of spaces"),
])
def test_clean_text(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize("bad", [None, 123, float("nan")])
def test_clean_text_non_string_returns_empty(bad):
    assert clean_text(bad) == ""


def test_clean_text_only_noise_becomes_empty():
    assert clean_text("http://a.b @x #y 123 !!!") == ""
