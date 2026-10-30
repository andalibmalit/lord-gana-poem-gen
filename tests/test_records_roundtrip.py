"""Poem files keep the poem text and metadata exactly, whatever the text contains."""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from poemgen import records

# Any Unicode text that UTF-8 can encode (no lone surrogates), biased towards the
# characters that break naive formats: newlines, front-matter markers, IAST.
tricky = st.sampled_from(["\n", "\r\n", "\r", "---", "\n---\n", "Oṃ Gaṃ", "ṛ", "﻿", " ", "#"])
text = st.lists(
    st.one_of(tricky, st.text(alphabet=st.characters(exclude_categories=("Cs",)))), max_size=12
).map("".join)


@settings(max_examples=300, deadline=None)
@given(poem=text, thinking=text, note=text)
def test_poem_round_trip(tmp_path_factory, poem, thinking, note):
    path = Path(tmp_path_factory.mktemp("p")) / "id.md"
    meta = {"poem_id": "id", "note": note, "nested": {"list": [note, 1.5, None]}}
    records.write_poem(path, meta, poem, thinking or None)
    loaded_meta, loaded_poem = records.load_poem(path)
    assert loaded_poem == poem
    assert loaded_meta == meta
    if thinking:
        assert records.read_text(records.thinking_path(path)) == thinking
