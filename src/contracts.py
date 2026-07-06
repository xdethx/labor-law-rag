"""Employment contract PDF parsing + clause splitting (M5).

Qdrant-free and model-free, like src/parsing.py — this module only turns
uploaded PDF bytes into a list of clause chunks. Embedding/upsert into the
`contracts` collection lives in src/retrieval.py; upload hardening
(extension/MIME/size checks) lives in src/main.py.
"""

import re
from io import BytesIO

from pypdf import PdfReader

# "1.", "1)", "1-", "Madde 1", "MADDE 1 -" etc. at the start of a line —
# contracts number their own articles/clauses in varied, informal styles.
CLAUSE_HEADING_RE = re.compile(r"^\s*(?:MADDE\s+)?(\d{1,3})\s*[.)\-]\s*\S", re.IGNORECASE)

# Below this many detected headings, numbered-clause splitting is unreliable
# (e.g. a contract with only one or two numbered articles) -> fall back to
# paragraph splitting instead of producing one giant "clause".
_MIN_HEADINGS_FOR_SPLIT = 2


class EncryptedPDFError(ValueError):
    """Raised when the uploaded PDF is password-protected."""


def extract_contract_text(pdf_bytes: bytes) -> tuple[str, int]:
    """Extract full text + page count from PDF bytes.

    Raises EncryptedPDFError if the PDF is password-protected — pypdf can
    still report a page count for these, but extract_text() returns
    empty/garbage, so we reject up front rather than upsert junk clauses.
    """
    reader = PdfReader(BytesIO(pdf_bytes))
    if reader.is_encrypted:
        raise EncryptedPDFError("uploaded PDF is encrypted/password-protected")
    text = "\n".join(page.extract_text() for page in reader.pages)
    return text, len(reader.pages)


def _to_clause_dicts(texts: list[str]) -> list[dict]:
    return [{"clause_no": i + 1, "text": t} for i, t in enumerate(texts)]


def _split_by_heading(lines: list[str]) -> list[dict] | None:
    """Group lines into clauses starting at each numbered heading. Returns
    None if too few headings were found (caller should fall back)."""
    starts = [i for i, line in enumerate(lines) if CLAUSE_HEADING_RE.match(line)]
    if len(starts) < _MIN_HEADINGS_FOR_SPLIT:
        return None

    bodies = []
    for idx, start in enumerate(starts):
        end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
        body = "\n".join(lines[start:end]).strip()
        if body:
            bodies.append(body)
    return _to_clause_dicts(bodies)


def _split_by_paragraph(text: str) -> list[dict]:
    """Fallback: split on blank lines. Coarser, but every contract has
    paragraph breaks even without numbered headings."""
    paragraphs = re.split(r"\n\s*\n", text)
    bodies = [p.strip() for p in paragraphs if p.strip()]
    return _to_clause_dicts(bodies)


def split_clauses(text: str) -> list[dict]:
    """Split contract text into clauses: `[{"clause_no": int, "text": str}]`.

    Numbered-heading regex first (the contract's own "Madde n" / "1." style
    articles); paragraph-split fallback when too few headings are found.
    `clause_no` is a sequential 1-based index of the chunk, independent of
    whatever numbering the contract itself uses.
    """
    clauses = _split_by_heading(text.splitlines())
    if clauses is not None:
        return clauses
    return _split_by_paragraph(text)
