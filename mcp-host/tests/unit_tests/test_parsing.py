from agent.parsing import (
    AnswerStream,
    FinalAnswer,
    Paper,
    content_to_text,
    dedupe_papers,
    extract_json_object,
    parse_final_answer,
)


def test_content_to_text_variants():
    assert content_to_text(None) == ""
    assert content_to_text("hi") == "hi"
    assert content_to_text({"text": "a"}) == "a"
    assert content_to_text([{"type": "text", "text": " a "}, "b", {"type": "image"}]) == "a\nb"


def test_extract_json_from_fenced_block_with_chatter():
    raw = 'Sure! ```json\n{"answer": "x {not json}", "papers": []}\n``` hope that helps'
    assert extract_json_object(raw) == {"answer": "x {not json}", "papers": []}


def test_extract_json_skips_broken_leading_brace():
    assert extract_json_object('{oops} then {"a": 1}') == {"a": 1}


def test_extract_json_none():
    assert extract_json_object("no json here") is None


def test_parse_final_answer_ok_and_drops_bad_items():
    raw = '{"answer": "A", "papers": [{"title": " p1.pdf "}, {"url": "x"}, "junk"], "recommended_papers": null}'
    final, ok = parse_final_answer(raw)
    assert ok
    assert final.answer == "A"
    assert final.papers == [Paper(title="p1.pdf", url="")]
    assert final.recommended_papers == []


def test_parse_final_answer_falls_back_to_raw_text():
    final, ok = parse_final_answer("  plain answer  ")
    assert not ok
    assert final == FinalAnswer(answer="plain answer")


def test_dedupe_is_case_insensitive_and_respects_exclude_and_limit():
    ps = [Paper(title="A"), Paper(title="a"), Paper(title="B"), Paper(title="C"), Paper(title="D")]
    assert [p.title for p in dedupe_papers(ps, exclude={"b"}, limit=2)] == ["A", "C"]


def test_answer_stream_hides_json_and_handles_split_escapes():
    reply = '{"answer": "Soft \\"targets\\"\\nfrom caf\\u00e9 models \\ud83d\\ude00", "sources": [1, 2]}'
    for step in (1, 3, 7):  # tokens can cut anywhere, including inside \" or é
        s = AnswerStream()
        out = "".join(s.feed(reply[i:i + step]) for i in range(0, len(reply), step))
        assert out == 'Soft "targets"\nfrom café models \U0001F600'


def test_answer_stream_waits_for_the_answer_key():
    s = AnswerStream()
    assert s.feed('{"ans') == ""
    assert s.feed('wer": "Hi') == "Hi"
    assert s.feed(' there", "sources": []}') == " there"
    assert s.feed("anything after the answer") == ""
