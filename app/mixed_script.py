"""Detection of identifiers that mix Latin, Greek and Cyrillic letters.

This is a companion to the NFKC collision scanner in :mod:`app.identifiers`:
it shares the exact lexical boundary (:func:`app.identifiers.iter_raw_identifiers`)
so comments, strings, numbers, operators and hard keywords are skipped the
same way (including NFKC-confusable keyword spellings), and every lexical
failure is reported with the same :class:`~app.identifiers.IdentifierLexError`
original-source position.

A *mixed-script* identifier is one single legal identifier whose letters
belong to at least two of the Latin, Greek and Cyrillic scripts.  Script
membership is taken literally from the Unicode character name: only letters
whose name starts with ``LATIN``, ``GREEK`` or ``CYRILLIC`` count.  This
deliberately ignores visual lookalikes encoded elsewhere (fullwidth and
mathematical letters, whose names start with ``FULLWIDTH`` /
``MATHEMATICAL``), combining marks (``COMBINING ...``), digits
(``... DIGIT ...``), underscores and every other continuation character --
they neither add a script nor break the identifier, exactly per the
detection rule.  Every relevant character keeps its own codepoint offset
and 1-based line/column in the original source.

Each mixed occurrence is reported separately, so the same disguised name
appearing twice yields two findings in source order.  Nothing is persisted
at this layer.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

from .identifiers import iter_raw_identifiers

# Script labels in first-occurrence order; the matching name prefixes are
# identical uppercase tokens.
_SCRIPTS = ("latin", "greek", "cyrillic")


def _script_of(ch: str) -> str | None:
    """Return ``'latin' | 'greek' | 'cyrillic'`` for *ch*, else ``None``.

    Only letters whose Unicode name literally starts with LATIN/GREEK/
    CYRILLIC count; combining marks, digits, underscores and letters from
    any other block (incl. FULLWIDTH/MATHEMATICAL lookalikes) return None.
    """
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return None
    for script in _SCRIPTS:
        if name.startswith(script.upper()):
            return script
    return None


@dataclass(frozen=True)
class ScriptedCharacter:
    """One relevant letter inside a mixed-script identifier."""

    character: str
    script: str
    codepoint: str     # "U+XXXX" (or U+XXXXX on the astral plane)
    offset: int        # codepoint offset in the original source (0-based)
    line: int          # 1-based source line
    column: int        # 1-based source column (codepoints)


@dataclass(frozen=True)
class MixedScriptIdentifier:
    """One occurrence of an identifier spanning two or more scripts."""

    spelling: str
    scripts: list[str]                 # distinct scripts, first-seen order
    offset: int                        # first character offset (0-based)
    end_offset: int                    # offset just past the last character
    line: int
    column: int
    characters: list[ScriptedCharacter]  # every relevant letter, in order


@dataclass(frozen=True)
class MixedScriptReport:
    identifier_count: int             # total non-keyword identifier occurrences
    mixed_script_count: int
    mixed_identifiers: list[MixedScriptIdentifier]

    def to_dict(self) -> dict:
        return {
            "identifier_count": self.identifier_count,
            "mixed_script_count": self.mixed_script_count,
            "mixed_identifiers": [
                {
                    "spelling": m.spelling,
                    "scripts": m.scripts,
                    "offset": m.offset,
                    "end_offset": m.end_offset,
                    "line": m.line,
                    "column": m.column,
                    "characters": [
                        {
                            "character": c.character,
                            "script": c.script,
                            "codepoint": c.codepoint,
                            "offset": c.offset,
                            "line": c.line,
                            "column": c.column,
                        }
                        for c in m.characters
                    ],
                }
                for m in self.mixed_identifiers
            ],
        }


def scan_mixed_scripts(source: str) -> MixedScriptReport:
    """Scan Python *source* for mixed Latin/Greek/Cyrillic identifiers.

    Raises :class:`~app.identifiers.IdentifierLexError` on any lexical
    failure.
    """
    findings: list[MixedScriptIdentifier] = []
    total_identifiers = 0

    for raw in iter_raw_identifiers(source):
        total_identifiers += 1
        occ = raw.occurrence
        scripts: list[str] = []
        characters: list[ScriptedCharacter] = []
        for i, ch in enumerate(raw.spelling):
            script = _script_of(ch)
            if script is None:
                # Combining marks, digits, underscores and letters outside
                # the three tracked scripts neither count nor separate.
                continue
            if script not in scripts:
                scripts.append(script)
            characters.append(
                ScriptedCharacter(
                    character=ch,
                    script=script,
                    codepoint=f"U+{ord(ch):04X}",
                    offset=occ.offset + i,
                    line=occ.line,
                    column=occ.column + i,
                )
            )
        if len(scripts) >= 2:
            findings.append(
                MixedScriptIdentifier(
                    spelling=raw.spelling,
                    scripts=scripts,
                    offset=occ.offset,
                    end_offset=occ.end_offset,
                    line=occ.line,
                    column=occ.column,
                    characters=characters,
                )
            )

    return MixedScriptReport(
        identifier_count=total_identifiers,
        mixed_script_count=len(findings),
        mixed_identifiers=findings,
    )
