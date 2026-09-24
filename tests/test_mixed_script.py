"""Tests for mixed-script identifier detection (POST identifiers/mixed-script).

Covers: reuse of the shared identifier lexical boundary (comments, strings
and hard keywords skipped, including NFKC-confusable full-width keyword
spellings; soft keywords kept), the LATIN/GREEK/CYRILLIC Unicode-name rule
(fullwidth/mathematical letters, combining marks, digits and underscores
do not count), two- and three-script mixes, per-occurrence reporting of
repeated names, original-source codepoint offsets/line/column for the
identifier and every relevant character, astral-safe spans, BOM/CRLF
handling, lexical failures returning the shared 422 structure, the API
contract (200, filename echo, not persisted) and the root listing.

Non-ASCII characters are written as \\uXXXX escapes so the file's exact
codepoint composition is independent of editor normalisation.
"""

import pytest

from app.identifiers import IdentifierLexError
from app.mixed_script import scan_mixed_scripts

LATIN_A = "a"
GREEK_ALPHA = "α"        # U+03B1 GREEK SMALL LETTER ALPHA
GREEK_OMEGA = "ω"        # U+03C9 GREEK SMALL LETTER OMEGA
CYR_A = "а"              # U+0430 CYRILLIC SMALL LETTER A
CYR_O = "о"              # U+043E CYRILLIC SMALL LETTER O
E_COMBINING = "e" + chr(0x0301)  # 'e' + combining acute U+0301
FW_X = "Ｘ"              # U+FF38 FULLWIDTH LATIN CAPITAL LETTER X
FW_TRUE = "Ｔｒｕｅ"  # fullwidth spelling that NFKC-normalises to keyword True
BOM = "﻿"


# ----------------------------------------------------------- unit: script --

def test_latin_cyrillic_mix_is_flagged():
    # 'sc' Latin + U+043E Cyrillic O + 're' Latin: classic homoglyph.
    src = f"score = 1\nsc{CYR_O}re = 2\n"
    report = scan_mixed_scripts(src).to_dict()
    assert report["identifier_count"] == 2
    assert report["mixed_script_count"] == 1
    mixed = report["mixed_identifiers"][0]
    assert src[mixed["offset"]:mixed["end_offset"]] == f"sc{CYR_O}re"
    assert mixed["scripts"] == ["latin", "cyrillic"]
    assert mixed["line"] == 2 and mixed["column"] == 1


def test_latin_greek_mix_is_flagged():
    report = scan_mixed_scripts(f"x{GREEK_OMEGA} = 1\n").to_dict()
    assert report["mixed_script_count"] == 1
    mixed = report["mixed_identifiers"][0]
    assert mixed["scripts"] == ["latin", "greek"]


def test_three_scripts_are_reported_in_first_seen_order():
    src = LATIN_A + GREEK_ALPHA + CYR_A + " = 1\n"
    mixed = scan_mixed_scripts(src).to_dict()["mixed_identifiers"][0]
    assert mixed["scripts"] == ["latin", "greek", "cyrillic"]


def test_pure_latin_greek_or_cyrillic_is_not_flagged():
    report = scan_mixed_scripts(
        f"plain = 1\n{GREEK_ALPHA}{GREEK_OMEGA} = 2\nшаблон = 3\n"
    ).to_dict()
    assert report["identifier_count"] == 3
    assert report["mixed_script_count"] == 0


def test_greek_cyrillic_without_latin_is_flagged():
    report = scan_mixed_scripts(f"{GREEK_ALPHA}{CYR_A} = 1\n").to_dict()
    assert report["mixed_script_count"] == 1
    assert report["mixed_identifiers"][0]["scripts"] == ["greek", "cyrillic"]


def test_characters_list_only_counts_latin_greek_cyrillic_letters():
    # digit, underscore and a combining mark must not appear in characters
    src = f"a1_{E_COMBINING}{GREEK_ALPHA} = 1\n"
    mixed = scan_mixed_scripts(src).to_dict()["mixed_identifiers"][0]
    chars = mixed["characters"]
    assert [c["character"] for c in chars] == ["a", "e", GREEK_ALPHA]
    assert [c["script"] for c in chars] == ["latin", "latin", "greek"]
    # codepoint annotations
    assert chars[1]["codepoint"] == "U+0065"
    assert chars[2]["codepoint"] == "U+03B1"


def test_fullwidth_and_mathematical_letters_do_not_count_as_latin():
    # U+FF38's name starts with FULLWIDTH, not LATIN; even next to a Greek
    # letter it must not look like a Latin/Greek mix.
    report = scan_mixed_scripts(f"{FW_X}{GREEK_OMEGA} = 1\n").to_dict()
    assert report["mixed_script_count"] == 0
    mixed_greek = report["mixed_identifiers"]
    assert mixed_greek == []
    # mathematical bold small a (U+1D41A, name MATHEMATICAL ...) + Cyrillic
    math_a = "𝐚"
    report2 = scan_mixed_scripts(f"{math_a}{CYR_A} = 1\n").to_dict()
    assert report2["mixed_script_count"] == 0


def test_combining_mark_alone_adds_no_script():
    # Greek letters plus a COMBINING ACUTE (name starts with COMBINING):
    # still single-script Greek.
    src = f"{GREEK_ALPHA}" + chr(0x0301) + f"{GREEK_OMEGA} = 1\n"
    report = scan_mixed_scripts(src).to_dict()
    assert report["mixed_script_count"] == 0


def test_same_spelling_reported_for_every_occurrence():
    name = f"sc{CYR_O}re"
    src = f"{name} = 1\n{name} = {name} + 1\n"
    report = scan_mixed_scripts(src).to_dict()
    assert report["mixed_script_count"] == 3
    offsets = [m["offset"] for m in report["mixed_identifiers"]]
    assert offsets == sorted(offsets)
    assert all(m["spelling"] == name for m in report["mixed_identifiers"])


def test_character_positions_match_original_source():
    src = f"x = sc{CYR_O}re\n"
    mixed = scan_mixed_scripts(src).to_dict()["mixed_identifiers"][0]
    assert mixed["offset"] == 4 and mixed["line"] == 1 and mixed["column"] == 5
    for c in mixed["characters"]:
        assert src[c["offset"]] == c["character"]
        assert c["line"] == 1
        assert c["column"] == c["offset"] + 1
    # the Cyrillic letter is the 3rd character of the name
    cyr = [c for c in mixed["characters"] if c["script"] == "cyrillic"][0]
    assert cyr["offset"] == 6 and cyr["column"] == 7
    assert cyr["codepoint"] == "U+043E"


def test_multiline_offsets_and_lines():
    src = f"a = 1\n\nx{GREEK_OMEGA} = 2\n"
    mixed = scan_mixed_scripts(src).to_dict()["mixed_identifiers"][0]
    assert src[mixed["offset"]:mixed["end_offset"]] == f"x{GREEK_OMEGA}"
    assert (mixed["line"], mixed["offset"]) == (3, 7)


def test_bom_is_accepted_and_offsets_include_it():
    src = f"{BOM}x{CYR_A} = 1\n"
    mixed = scan_mixed_scripts(src).to_dict()["mixed_identifiers"][0]
    assert mixed["offset"] == 1 and mixed["column"] == 2
    assert src[mixed["offset"]:mixed["end_offset"]] == f"x{CYR_A}"
    first, second = mixed["characters"]
    assert (first["offset"], first["column"]) == (1, 2)
    assert (second["offset"], second["column"]) == (2, 3)


def test_crlf_columns_match_codepoints():
    src = f"x{GREEK_OMEGA} = 1\r\na{CYR_A} = 2\r\n"
    report = scan_mixed_scripts(src).to_dict()
    for mixed in report["mixed_identifiers"]:
        assert src[mixed["offset"]:mixed["end_offset"]] == mixed["spelling"]
        for c in mixed["characters"]:
            assert src[c["offset"]] == c["character"]


def test_empty_and_comment_only_source():
    assert scan_mixed_scripts("").to_dict() == {
        "identifier_count": 0,
        "mixed_script_count": 0,
        "mixed_identifiers": [],
    }
    report = scan_mixed_scripts(f"# sc{CYR_O}re\n").to_dict()
    assert report["identifier_count"] == 0
    assert report["mixed_script_count"] == 0


# ----------------------------------------------------- unit: shared lexing --

def test_comments_and_strings_are_skipped():
    src = (
        f"# x{CYR_A} comment\n"
        f"s = 'x{CYR_A} inside'\n"
        f't = f"{{x{GREEK_ALPHA}}}"\n'
        f'doc = """x{CYR_A}"""\n'
        f"plain = 1\n"
    )
    report = scan_mixed_scripts(src).to_dict()
    # s, t, doc, plain survive; nothing inside comments/strings contributes
    assert report["identifier_count"] == 4
    assert report["mixed_script_count"] == 0


def test_keywords_are_skipped_including_confusable_spellings():
    # Full-width letters NFKC-normalise to the hard keyword 'True'; the
    # mixed-script scanner must skip it just like the collision scanner.
    report = scan_mixed_scripts(
        f"{FW_TRUE} = 1\nif {FW_TRUE}: pass\n"
    ).to_dict()
    assert report["identifier_count"] == 0
    assert report["mixed_script_count"] == 0


def test_soft_keywords_remain_identifiers():
    report = scan_mixed_scripts(f"mаtch = 1\n").to_dict()
    assert report["identifier_count"] == 1
    assert report["mixed_script_count"] == 1


def test_mixed_findings_are_in_source_order():
    first = f"a{GREEK_ALPHA}"
    second = f"b{CYR_A}"
    third = f"c{GREEK_OMEGA}"
    report = scan_mixed_scripts(
        f"x = 1\n{second} = 2\n{first} = 3\n{third} = 4\n"
    ).to_dict()
    assert [m["spelling"] for m in report["mixed_identifiers"]] == [
        second, first, third
    ]


# ------------------------------------------------------- unit: lex errors --

@pytest.mark.parametrize(
    "source,error",
    [
        ("x = 1 $\n", "unexpected_character"),
        ("x\x00y = 1\n", "unexpected_character"),
        ("x = '''unterminated\n", "unterminated_string"),
        ("x = '", "unterminated_statement"),
        ("def f():\n  x=1\n y=2\n", "indentation_error"),
    ],
)
def test_lexical_failure_raises_with_position(source, error):
    with pytest.raises(IdentifierLexError) as exc:
        scan_mixed_scripts(source)
    assert exc.value.error == error
    assert exc.value.line >= 1
    assert exc.value.column >= 1
    assert 0 <= exc.value.offset <= len(source)


# ---------------------------------------------------------------- API: POST -

def test_post_mixed_script_shape(client):
    src = f"score = 1\nsc{CYR_O}re = 2\n"
    resp = client.post(
        "/api/v1/identifiers/mixed-script",
        json={"source": src, "filename": "m.py"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["stored"] is False
    assert body["filename"] == "m.py"
    assert body["source_length"] == len(src)
    report = body["report"]
    assert report["identifier_count"] == 2
    assert report["mixed_script_count"] == 1
    mixed = report["mixed_identifiers"][0]
    assert mixed["spelling"] == f"sc{CYR_O}re"
    assert mixed["scripts"] == ["latin", "cyrillic"]
    cyr = [c for c in mixed["characters"] if c["script"] == "cyrillic"]
    assert len(cyr) == 1 and cyr[0]["codepoint"] == "U+043E"
    assert "id" not in body


def test_post_clean_source_empty_report(client):
    resp = client.post(
        "/api/v1/identifiers/mixed-script",
        json={"source": "alpha = 1\n"},
    )
    assert resp.status_code == 200, resp.text
    report = resp.json()["report"]
    assert report["identifier_count"] == 1
    assert report["mixed_script_count"] == 0
    assert report["mixed_identifiers"] == []


def test_post_filename_optional_and_echoed(client):
    resp = client.post(
        "/api/v1/identifiers/mixed-script", json={"source": f"x{GREEK_ALPHA} = 1\n"}
    )
    assert resp.status_code == 200
    assert resp.json()["filename"] is None


def test_post_lex_failure_422_same_structure(client):
    resp = client.post(
        "/api/v1/identifiers/mixed-script", json={"source": "x = 1 $\n"}
    )
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    assert detail["error"] == "unexpected_character"
    assert detail["offset"] == 6
    assert detail["line"] == 1 and detail["column"] == 7
    assert "message" in detail


@pytest.mark.parametrize(
    "source,error",
    [
        ("x = '''never ends\n", "unterminated_string"),
        ("x = '", "unterminated_statement"),
        ("def f():\n  x=1\n y=2\n", "indentation_error"),
    ],
)
def test_post_other_lex_failures_422(client, source, error):
    resp = client.post(
        "/api/v1/identifiers/mixed-script", json={"source": source}
    )
    assert resp.status_code == 422
    assert resp.json()["detail"]["error"] == error


def test_post_missing_source_field_422(client):
    resp = client.post(
        "/api/v1/identifiers/mixed-script", json={"filename": "m.py"}
    )
    assert resp.status_code == 422


def test_post_writes_no_records(client):
    client.post(
        "/api/v1/identifiers/mixed-script",
        json={"source": f"sc{CYR_O}re = 1\n"},
    )
    # neither table receives a row
    assert client.get("/api/v1/audits").json()["total"] == 0
    # identifier reports have their own independent sequence; a stored
    # normalization report created afterwards still starts at id 1
    created = client.post(
        "/api/v1/identifiers/normalization", json={"source": "y = 1\n"}
    ).json()
    assert created["id"] == 1


def test_root_lists_mixed_script_endpoint(client):
    endpoints = client.get("/").json()["endpoints"]
    assert "POST /api/v1/identifiers/mixed-script" in endpoints
