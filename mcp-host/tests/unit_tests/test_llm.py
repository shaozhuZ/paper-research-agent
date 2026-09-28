from dataclasses import replace

import pytest

from agent.config import settings
from agent.llm import make_llm


@pytest.fixture(autouse=True)
def fake_keys(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")


def test_google():
    llm = make_llm(replace(settings, llm_model="google:gemini-3-flash-preview"))
    assert type(llm).__name__ == "ChatGoogleGenerativeAI"


def test_openai_passes_reasoning_effort_but_not_temperature():
    llm = make_llm(replace(settings, llm_model="openai:gpt-6-luna", reasoning_effort="low"))
    assert type(llm).__name__ == "ChatOpenAI"
    assert llm.model_name == "gpt-6-luna"
    assert llm.reasoning_effort == "low"
    assert llm.temperature is None


@pytest.mark.parametrize("bad", ["gpt-6-luna", "mistral:large", "openai:"])
def test_rejects_bad_model_strings(bad):
    with pytest.raises(ValueError):
        make_llm(replace(settings, llm_model=bad))
