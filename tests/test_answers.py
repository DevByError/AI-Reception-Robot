"""Stage 05: answer chain order, streamed general answers, prefetch."""

import dialogue_manager as dm
import dialogue_manager_live as dml
from knowledge_base import KnowledgeBase, KnowledgeChunk


class _Fake:
    def __init__(self, name, general=None, fail=False, grounded="It is on the third floor."):
        self.name, self.general, self.fail, self.grounded = name, general, fail, grounded
        self.calls = 0

    def health_check(self):
        if self.fail:
            raise RuntimeError("503 model overloaded")

    def generate_grounded_answer(self, q, chunks):
        if self.fail:
            raise RuntimeError("503 model overloaded")
        return self.grounded

    def stream_grounded_answer(self, q, chunks):
        if self.fail:
            raise RuntimeError("503 model overloaded")
        yield from self.grounded.split(" ")

    def classify_and_answer_general(self, q):
        if self.fail:
            raise RuntimeError("503")
        return False, "fine thanks"

    def stream_general_answer(self, q):
        self.calls += 1
        if self.fail:
            raise RuntimeError("429 quota exhausted")
        yield from self.general


def test_header_is_stripped_and_label_parsed():
    info = {}
    out = "".join(
        dm.strip_general_header(
            iter(["UNIVERSITY_", "RELATED: no\nANS", "WER: Sure, ", "it's sunny."]), info
        )
    )
    assert out == "Sure, it's sunny."
    assert info["university_related"] is False and info["answered"] is True


def test_model_ignoring_the_format_is_still_spoken():
    info = {}
    text = "Honestly I am not sure, but you could check the website for that. " * 3
    out = "".join(dm.strip_general_header(iter([text]), info))
    assert out.strip().startswith("Honestly") and info["answered"]


def test_blank_answer_marks_unanswered():
    info = {}
    out = "".join(dm.strip_general_header(iter(["UNIVERSITY_RELATED: yes\nANSWER: "]), info))
    assert out == "" and info["answered"] is False and info["university_related"] is True


def _kb():
    kb = KnowledgeBase()
    kb.upsert(
        KnowledgeChunk(
            "lib", "The library is on the second floor.", "x", 0.0, is_direct_answer=True
        )
    )
    return kb


def test_general_questions_stream_and_fall_back_when_blank():
    gen = _Fake("a", general=["UNIVERSITY_RELATED: no\nANSWER: ", "Sure! ", "I'm doing well."])
    manager = dm.DialogueManager(_kb(), gen)
    result = manager.handle_question_stream("how are you today")
    assert result.is_streaming
    assert "".join(result.pieces) == "Sure! I'm doing well."
    assert result.intent == dm.Intent.GENERAL_CHAT

    blank = _Fake("b", general=["UNIVERSITY_RELATED: yes\nANSWER:"])
    result = dm.DialogueManager(_kb(), blank).handle_question_stream("what's the cafeteria menu")
    assert "".join(result.pieces) == dm.FALLBACK_ANSWER
    assert result.intent == dm.Intent.UNKNOWN


def test_streaming_general_can_be_switched_off():
    gen = _Fake("a", general=["UNIVERSITY_RELATED: no\nANSWER: hi"])
    result = dm.DialogueManager(_kb(), gen, stream_general=False).handle_question_stream("hello")
    assert not result.is_streaming and "".join(result.pieces) == "fine thanks"


def test_chain_fails_over_before_first_word_for_general_answers():
    down = _Fake("down", fail=True)
    up = _Fake("up", general=["UNIVERSITY_RELATED: no\nANSWER: hello there"])
    chain = dml.ChainAnswerGenerator([("Groq", down), ("Ollama", up)])
    assert "".join(chain.stream_general_answer("hi")).endswith("hello there")
    assert down.calls == 1 and up.calls == 1
    chain.close()


def test_backend_order_follows_config(monkeypatch):
    made = []
    monkeypatch.setattr(
        dml,
        "_CLOUD_BACKENDS",
        {
            "groq": ("Groq", "GROQ_API_KEY", lambda: made.append("groq") or _Fake("groq")),
            "gemini": (
                "Gemini",
                "GEMINI_API_KEY",
                lambda: made.append("gemini") or _Fake("gemini"),
            ),
            "claude": ("Claude", "ANTHROPIC_API_KEY", lambda: _Fake("claude")),
        },
    )
    monkeypatch.setenv("GROQ_API_KEY", "x")
    monkeypatch.setenv("GEMINI_API_KEY", "y")
    monkeypatch.setattr(dml, "_build_local_generator", lambda warm=True: ("Ollama", _Fake("o")))
    monkeypatch.setattr(dml.RT, "ANSWER_BACKENDS", "groq,gemini,ollama")
    chain = dml._build_answer_generator()
    assert chain.names == ["Groq", "Gemini", "Ollama"]
    chain.close()
    monkeypatch.setattr(dml.RT, "ANSWER_BACKENDS", "ollama,gemini")
    chain = dml._build_answer_generator()
    assert chain.names == ["Ollama", "Gemini"]
    chain.close()


def test_prefetch_is_used_only_for_the_same_words():
    kb = _kb()
    manager = dm.DialogueManager(kb, _Fake("a", general=["ANSWER: x"]))
    calls = []
    real = kb.search
    kb.search = lambda q: calls.append(q) or real(q)
    manager.prefetch("Where is the library?")
    assert len(calls) == 1
    result = manager.handle_question_stream("where is the library")
    assert "".join(result.pieces) == "The library is on the second floor."
    assert len(calls) == 1, "the final question should reuse the prefetched search"
    manager.prefetch("where is the libr")
    manager.handle_question_stream("where is the library")
    assert len(calls) == 3, "different words must be searched again"
