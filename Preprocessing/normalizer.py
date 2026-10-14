"""
normalizer.py
=============
Bengali Unicode canonicalization, Nukta/Khanda-Ta normalization, OCR/extraction
noise removal and whitespace cleanup for the Kothon dataset pipeline.

The central class, :class:`BengaliTextNormalizer`, is stateless with respect
to the *text* it processes but stateful with respect to an internal
*anomaly-count audit trail* it accumulates across calls to :meth:`normalize`.
That audit trail feeds Visualization Figure 5 (Character & Unicode Anomaly
Audit) in ``visualizer.py``.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class NormalizationError(Exception):
    """Raised when normalization fails on an unrecoverable input."""


# Static Unicode tables

# Decomposed consonant + nukta (U+09BC) -> canonical precomposed character.
# Note: per the spec, decomposed forms are (consonant, U+09BC). We keep the
# mapping explicit and readable rather than derived, since only three
# Bengali nukta letters exist in modern orthography.
_NUKTA = "\u09BC"
_NUKTA_MAP: Dict[str, str] = {
    "\u09A1" + _NUKTA: "\u09DC",  # ড + ় -> ড়
    "\u09A2" + _NUKTA: "\u09DD",  # ঢ + ় -> ঢ়
    "\u09AF" + _NUKTA: "\u09DF",  # য + ় -> য়
}
_NUKTA_PATTERN = re.compile("|".join(re.escape(k) for k in _NUKTA_MAP))

# Khanda-Ta: ত (U+09A4) + হসন্ত (U+09CD) at a word boundary -> ৎ (U+09CE)
_KHANDA_TA_PATTERN = re.compile(r"\u09A4\u09CD(?=\s|$|[\u0964\u0965.,;:!?)\]\}])")
_KHANDA_TA_REPLACEMENT = "\u09CE"

# Dari / pipe normalization: ASCII pipe or Devanagari danda -> Bengali Dari.
_DARI_PATTERN = re.compile(r"[\u007C\u0964]")
_CANONICAL_DARI = "\u0964"  # Bengali/Devanagari share this codepoint; kept
# explicit so downstream readers see intent, even though the codepoint value
# is identical to Devanagari Danda (U+0964 is the shared Purna Chheda).

# Hyphenated line-break reconnection, e.g. "কর-\n তে" -> "করতে"
_HYPHEN_LINEBREAK_PATTERN = re.compile(r"(\w+)[-\u00AD]\s+(\w+)")

# Orphan / detached vowel matra: a dependent vowel sign with no preceding
# consonant/base character (typically caused by OCR dropping the base glyph
# or a stray leading matra after whitespace/newline).
_BENGALI_MATRAS = "\u09BE\u09BF\u09C0\u09C1\u09C2\u09C3\u09C4\u09C7\u09C8\u09CB\u09CC"
_ORPHAN_MATRA_PATTERN = re.compile(rf"(?<=\s)([{_BENGALI_MATRAS}])")
_ORPHAN_MATRA_LEADING_PATTERN = re.compile(rf"^([{_BENGALI_MATRAS}])")

# Zero-width joiner / non-joiner at word edges (stray insertion), while
# leaving *interior* ZWJ/ZWNJ alone since those are often linguistically
# meaningful (e.g. conjunct control in Bengali typography).
_ZW_LEADING_PATTERN = re.compile(r"^[\u200C\u200D]+")
_ZW_TRAILING_PATTERN = re.compile(r"[\u200C\u200D]+$")
_ZW_AROUND_WHITESPACE_PATTERN = re.compile(r"[\u200C\u200D]+(?=\s)|(?<=\s)[\u200C\u200D]+")

# Control characters to strip: \x00-\x1F except \n (\x0A) and \t (\x09).
_CONTROL_CHAR_PATTERN = re.compile(
    "[" + "".join(chr(c) for c in range(0x00, 0x20) if c not in (0x09, 0x0A)) + "]"
)

# Whitespace collapse: spaces, tabs, non-breaking space -> single space.
_HSPACE_PATTERN = re.compile(r"[ \t\u00A0]+")
_MULTI_NEWLINE_PATTERN = re.compile(r"\n{2,}")


class BengaliTextNormalizer:
    """Normalizes Bengali (and mixed Bengali/English) text extracted from
    novels, scanned books and news corpora into a clean, canonical form.

    Every call to :meth:`normalize` updates an internal
    :class:`collections.Counter` audit trail (see :attr:`anomaly_counts`)
    tracking how many times each anomaly class was fixed. This supports the
    "Character & Unicode Anomaly Audit" visualization (Figure 5).

    Parameters
    ----------
    track_anomalies:
        If True (default), increments internal counters for each anomaly
        class encountered. Disable for a small perf gain on very large,
        repeated batch calls where the audit trail is not needed.
    """

    def __init__(self, track_anomalies: bool = True) -> None:
        self.track_anomalies = track_anomalies
        self.anomaly_counts: Counter = Counter()

    # Public API
    def normalize(self, text: Optional[str]) -> str:
        """Run the full normalization pipeline on a single string.

        Returns an empty string for ``None``/non-string input rather than
        raising, since ingestion-level null handling is the responsibility
        of :mod:`preprocessor`.
        """
        if text is None:
            return ""
        if not isinstance(text, str):
            try:
                text = str(text)
            except Exception as exc:  # pragma: no cover - defensive
                raise NormalizationError(f"Cannot coerce value to str: {exc}") from exc

        text = self._strip_control_chars(text)
        text = self._unicode_nfc(text)
        text = self._reconnect_hyphenated_linebreaks(text)
        text = self._unify_nukta(text)
        text = self._normalize_khanda_ta(text)
        text = self._normalize_dari(text)
        text = self._fix_orphan_matras(text)
        text = self._strip_stray_zero_width(text)
        text = self._cleanup_whitespace(text)
        return text

    def normalize_series(self, series) -> "object":
        """Vectorized-friendly helper: normalize a pandas Series of text.

        Uses ``.map`` (still row-wise under the hood, since Unicode-aware
        regex normalization has no true vectorized equivalent in pandas/
        numpy) but keeps a single, explicit, well-logged entry point for
        callers in :mod:`preprocessor`.
        """
        import pandas as pd  # local import keeps this module import-light

        if not isinstance(series, pd.Series):
            raise TypeError(f"Expected pandas Series, got {type(series)!r}")
        return series.map(self.normalize)

    def reset_audit_trail(self) -> None:
        self.anomaly_counts.clear()

    def audit_report(self) -> Dict[str, int]:
        return dict(self.anomaly_counts)

    
    # Individual normalization steps
    def _count(self, key: str, n: int) -> None:
        if self.track_anomalies and n:
            self.anomaly_counts[key] += n

    def _strip_control_chars(self, text: str) -> str:
        matches = len(_CONTROL_CHAR_PATTERN.findall(text))
        self._count("control_chars_stripped", matches)
        return _CONTROL_CHAR_PATTERN.sub("", text)

    def _unicode_nfc(self, text: str) -> str:
        normalized = unicodedata.normalize("NFC", text)
        if normalized != text:
            self._count("nfc_normalized_chars", sum(1 for a, b in zip(text, normalized) if a != b) or 1)
        return normalized

    def _reconnect_hyphenated_linebreaks(self, text: str) -> str:
        matches = len(_HYPHEN_LINEBREAK_PATTERN.findall(text))
        self._count("hyphenated_linebreaks_reconnected", matches)
        return _HYPHEN_LINEBREAK_PATTERN.sub(r"\1\2", text)

    def _unify_nukta(self, text: str) -> str:
        matches = len(_NUKTA_PATTERN.findall(text))
        self._count("nukta_decompositions_unified", matches)
        return _NUKTA_PATTERN.sub(lambda m: _NUKTA_MAP[m.group(0)], text)

    def _normalize_khanda_ta(self, text: str) -> str:
        matches = len(_KHANDA_TA_PATTERN.findall(text))
        self._count("khanda_ta_normalized", matches)
        return _KHANDA_TA_PATTERN.sub(_KHANDA_TA_REPLACEMENT, text)

    def _normalize_dari(self, text: str) -> str:
        # Only count ASCII-pipe substitutions distinctly (the Devanagari
        # Danda already maps to the identical target codepoint).
        pipe_matches = text.count("\u007C")
        self._count("ascii_pipe_to_dari", pipe_matches)
        return _DARI_PATTERN.sub(_CANONICAL_DARI, text)

    def _fix_orphan_matras(self, text: str) -> str:
        # Strategy: an orphan matra (preceded by whitespace or at string
        # start, i.e. with no base consonant to attach to) is most often an
        # OCR artifact where the base glyph was dropped. We cannot safely
        # *guess* the missing consonant, so we normalize the anomaly by
        # removing the detached matra while flagging it in the audit trail
        # for manual review via the audit log (Phase 3 diagnostic log).
        n1 = len(_ORPHAN_MATRA_PATTERN.findall(text))
        text = _ORPHAN_MATRA_PATTERN.sub("", text)
        n2 = len(_ORPHAN_MATRA_LEADING_PATTERN.findall(text))
        text = _ORPHAN_MATRA_LEADING_PATTERN.sub("", text)
        self._count("orphan_matras_removed", n1 + n2)
        return text

    def _strip_stray_zero_width(self, text: str) -> str:
        n_lead = len(_ZW_LEADING_PATTERN.findall(text))
        text = _ZW_LEADING_PATTERN.sub("", text)
        n_trail = len(_ZW_TRAILING_PATTERN.findall(text))
        text = _ZW_TRAILING_PATTERN.sub("", text)
        n_ws = len(_ZW_AROUND_WHITESPACE_PATTERN.findall(text))
        text = _ZW_AROUND_WHITESPACE_PATTERN.sub("", text)
        # Interior ZWJ/ZWNJ (surrounded by non-whitespace on both sides) are
        # preserved intentionally -- they can be linguistically valid
        # conjunct-forming controls in Bengali typography.
        self._count("stray_zwj_zwnj_stripped", n_lead + n_trail + n_ws)
        return text

    def _cleanup_whitespace(self, text: str) -> str:
        text = _HSPACE_PATTERN.sub(" ", text)
        text = _MULTI_NEWLINE_PATTERN.sub("\n", text)
        return text.strip()
