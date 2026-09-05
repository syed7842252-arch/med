"""
MedLens Backend — backend/app.py
=================================

Evidence-First Clinical Information Intelligence
"AI extracts. Rules validate. Evidence explains. Humans verify."

SINGLE-FILE BACKEND. Everything — DB schema, PDF extraction, AI extraction
abstraction (with deterministic offline fallback), reference-range engine,
conflict detection, verification, timeline, summary generation, and the
REST API — lives in this file, as required by the architecture spec.

NOTE ON FRAMEWORK: the target environment for this build has no outbound
network access, so FastAPI/uvicorn could not be installed. This file is
implemented on Flask (stdlib-adjacent, already available) with an
equivalent route structure, equivalent request/response JSON shapes, and
the same SQLite persistence model. Every endpoint in the spec exists here
under the same path and method. If deployed somewhere with FastAPI
available, this is straightforward to port 1:1 (each @app.route already
maps to one FastAPI path operation) — see README.md.

Run:
    cd backend
    pip install -r requirements.txt
    python app.py
    # serves API on http://localhost:8000
"""

from __future__ import annotations

import io
import os
import re
import json
import socket
import sqlite3
import threading
import time
import uuid
import webbrowser
import datetime
from contextlib import contextmanager

from flask import Flask, request, jsonify, g, send_from_directory
from werkzeug.utils import secure_filename

try:
    from pypdf import PdfReader
    PYPDF_AVAILABLE = True
except Exception:  # pragma: no cover
    PYPDF_AVAILABLE = False

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _detect_frontend_dir():
    """Find index.html no matter which of the two common layouts is used,
    so this works right after a fresh `git clone` with zero manual setup —
    no env var, no folder rearranging required.

    Checked in order:
      1. MEDLENS_FRONTEND_DIR env var, if explicitly set (highest priority,
         still supported for unusual/custom layouts).
      2. Sibling "frontend/" folder next to this file's folder, e.g.
         MedLens/backend/app.py + MedLens/frontend/index.html
      3. The same folder this file lives in (flat layout), e.g.
         MED/app.py + MED/index.html
    Falls back to the sibling-folder path (option 2) if index.html isn't
    found anywhere, so the JSON fallback message in root() still reports
    a sensible, explainable location.
    """
    env_override = os.environ.get("MEDLENS_FRONTEND_DIR")
    if env_override:
        return env_override

    sibling_frontend = os.path.join(os.path.dirname(BASE_DIR), "frontend")
    if os.path.isfile(os.path.join(sibling_frontend, "index.html")):
        return sibling_frontend

    if os.path.isfile(os.path.join(BASE_DIR, "index.html")):
        return BASE_DIR

    return sibling_frontend  # not found anywhere; used for the error message


FRONTEND_DIR = _detect_frontend_dir()
DB_PATH = os.environ.get("MEDLENS_DB_PATH", os.path.join(BASE_DIR, "medlens.db"))
UPLOAD_DIR = os.environ.get("MEDLENS_UPLOAD_DIR", os.path.join(BASE_DIR, "uploads"))
MAX_UPLOAD_BYTES = int(os.environ.get("MEDLENS_MAX_UPLOAD_MB", "15")) * 1024 * 1024
ALLOWED_EXTENSIONS = {".pdf"}

# AI provider config — server-side ONLY. Never exposed to the frontend.
# If no key is configured, the app runs entirely on the deterministic
# offline extraction fallback (this is also what powers the demo & tests,
# so the product is fully functional with zero external dependency).
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "").strip()
AI_ENABLED = bool(ANTHROPIC_API_KEY)

os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES


# ---------------------------------------------------------------------------
# Database layer
# ---------------------------------------------------------------------------

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    age INTEGER,
    sex TEXT,
    symptoms TEXT,           -- JSON list
    conditions TEXT,         -- JSON list
    allergies TEXT,          -- JSON list
    medications TEXT,        -- JSON list
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    stored_path TEXT,
    report_date TEXT,        -- best-effort extracted / user supplied date
    status TEXT NOT NULL,    -- UPLOADED, EXTRACTED, EXTRACTION_FAILED
    error_message TEXT,
    uploaded_at TEXT NOT NULL,
    FOREIGN KEY (patient_id) REFERENCES patients(id)
);

CREATE TABLE IF NOT EXISTS document_pages (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    page_number INTEGER NOT NULL,
    text TEXT,
    FOREIGN KEY (document_id) REFERENCES documents(id)
);

CREATE TABLE IF NOT EXISTS patient_facts (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    category TEXT NOT NULL,      -- ALLERGY, CONDITION, MEDICATION, SYMPTOM
    value TEXT NOT NULL,
    origin TEXT NOT NULL,        -- USER_PROVIDED, REPORT_EXTRACTED, AI_GENERATED, CONFLICT, UNVERIFIED, VERIFIED
    verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED',  -- UNVERIFIED, VERIFIED, REJECTED
    document_id TEXT,
    page_number INTEGER,
    source_text TEXT,
    confidence REAL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    superseded_by TEXT
);

CREATE TABLE IF NOT EXISTS lab_results (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    document_id TEXT,
    test_name TEXT NOT NULL,
    value REAL,
    raw_value TEXT,
    unit TEXT,
    ref_low REAL,
    ref_high REAL,
    ref_raw TEXT,
    classification TEXT NOT NULL,   -- LOW, NORMAL, HIGH, NOT_CLASSIFIED
    report_date TEXT,
    page_number INTEGER,
    source_text TEXT,
    confidence REAL,
    origin TEXT NOT NULL DEFAULT 'REPORT_EXTRACTED',
    verification_state TEXT NOT NULL DEFAULT 'UNVERIFIED',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS evidence (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    subject_type TEXT NOT NULL,   -- LAB_RESULT, PATIENT_FACT
    subject_id TEXT NOT NULL,
    document_id TEXT,
    page_number INTEGER,
    source_text TEXT,
    method TEXT,                  -- e.g. "Deterministic reference-range comparison"
    confidence REAL,
    origin TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS conflicts (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    category TEXT NOT NULL,       -- ALLERGY, CONDITION, MEDICATION, LAB
    field_label TEXT NOT NULL,
    current_value TEXT NOT NULL,
    current_fact_id TEXT,
    previous_value TEXT NOT NULL,
    previous_fact_id TEXT,
    status TEXT NOT NULL DEFAULT 'REQUIRES_HUMAN_VERIFICATION',
    resolution TEXT,              -- null, or the resolved value chosen by a human
    created_at TEXT NOT NULL,
    resolved_at TEXT
);

CREATE TABLE IF NOT EXISTS verification_events (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    subject_type TEXT NOT NULL,   -- PATIENT_FACT, LAB_RESULT, CONFLICT
    subject_id TEXT NOT NULL,
    action TEXT NOT NULL,         -- VERIFY, EDIT, REJECT
    previous_value TEXT,
    new_value TEXT,
    note TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS timeline_events (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    event_type TEXT NOT NULL,    -- REPORT_UPLOADED, LAB_RESULT, PATIENT_FACT, VERIFICATION, CONFLICT
    event_date TEXT NOT NULL,
    title TEXT NOT NULL,
    detail TEXT,                 -- JSON blob with event-specific structured detail
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS summaries (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL,
    text TEXT NOT NULL,
    generated_by TEXT NOT NULL,  -- AI_MODEL or OFFLINE_TEMPLATE
    created_at TEXT NOT NULL
);
"""


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
        g.db.row_factory = sqlite3.Row
        g.db.execute("PRAGMA foreign_keys = ON")
    return g.db


@app.teardown_appcontext
def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


@contextmanager
def db_conn_standalone():
    """A standalone connection for use outside a Flask request context
    (e.g. from test scripts)."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def now_iso():
    return datetime.datetime.utcnow().isoformat() + "Z"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def row_to_dict(row):
    return dict(row) if row is not None else None


def rows_to_list(rows):
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# REFERENCE RANGE ENGINE — deterministic, no LLM involvement whatsoever.
# ---------------------------------------------------------------------------

def classify_value(value, ref_low, ref_high) -> str:
    """
    Pure deterministic classification. The LLM never touches this decision.

        value < ref_low   -> LOW
        value > ref_high  -> HIGH
        otherwise         -> NORMAL

    If value or a usable reference range is missing -> NOT_CLASSIFIED.
    """
    if value is None or ref_low is None or ref_high is None:
        return "NOT_CLASSIFIED"
    try:
        value = float(value)
        ref_low = float(ref_low)
        ref_high = float(ref_high)
    except (TypeError, ValueError):
        return "NOT_CLASSIFIED"
    if ref_low > ref_high:
        # malformed reference range from source — refuse to guess
        return "NOT_CLASSIFIED"
    if value < ref_low:
        return "LOW"
    if value > ref_high:
        return "HIGH"
    return "NORMAL"


def parse_reference_range(raw: str):
    """Parse a reference-range string like '12-16', '12 - 16', '12–16' into
    (low, high) floats. Returns (None, None) if it cannot be parsed safely.
    This NEVER falls back to general medical knowledge — if the source text
    doesn't contain a parseable numeric range, we report NOT_CLASSIFIED."""
    if not raw:
        return None, None
    raw = raw.strip()
    m = re.match(r"^\s*(-?\d+(?:\.\d+)?)\s*[-–—to]+\s*(-?\d+(?:\.\d+)?)\s*$", raw, re.IGNORECASE)
    if m:
        return float(m.group(1)), float(m.group(2))
    # "< 5.0" style upper-bound-only ranges: treat low as None (NOT_CLASSIFIED
    # unless we can safely determine both bounds — we do NOT invent a low bound)
    return None, None


# ---------------------------------------------------------------------------
# PDF PROCESSING
# ---------------------------------------------------------------------------

class PdfExtractionError(Exception):
    pass


def extract_pdf_pages(file_bytes: bytes):
    """Return a list of {page_number, text} dicts. Raises PdfExtractionError
    on unreadable/invalid/empty files. Never crashes the caller."""
    if not file_bytes:
        raise PdfExtractionError("Uploaded file is empty.")
    if not PYPDF_AVAILABLE:
        raise PdfExtractionError("PDF library unavailable on server.")
    try:
        reader = PdfReader(io.BytesIO(file_bytes))
    except Exception as e:
        raise PdfExtractionError(f"Invalid or corrupt PDF: {e}")

    if len(reader.pages) == 0:
        raise PdfExtractionError("PDF contains no pages.")

    pages = []
    any_text = False
    for i, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        if text.strip():
            any_text = True
        pages.append({"page_number": i + 1, "text": text})

    if not any_text:
        # No selectable text — likely a scanned PDF. We note this rather
        # than pretending OCR happened (no OCR engine bundled in this
        # offline-first build); the caller gets pages with empty text and
        # a status the frontend can surface clearly.
        pass

    return pages, any_text


# ---------------------------------------------------------------------------
# AI EXTRACTION ABSTRACTION
#
# Two implementations behind one interface:
#   - call_ai_extractor(): uses the Anthropic API if ANTHROPIC_API_KEY is set
#   - offline_extractor(): fully deterministic regex/heuristic extraction,
#     used whenever no API key is configured (default for this build/demo)
#
# BOTH implementations obey the same strict rules:
#   never invent values / units / reference ranges / dates / pages;
#   return null when unavailable; preserve source text verbatim.
# ---------------------------------------------------------------------------

LAB_LINE_PATTERN = re.compile(
    r"""^\s*
    (?P<name>[A-Za-z][A-Za-z0-9 /()%\-]{1,40}?)      # test name
    \s*:?\s+                                          # optional trailing colon after name (e.g. "WBC:")
    (?P<value>-?\d+(?:\.\d+)?)                        # numeric value
    \s*
    (?P<unit>[A-Za-z%/µμ\^\*0-9.]{1,20})?             # unit (optional); allow ^ and * for units like x10^9/L
    \s*
    \(?\s*
    (?P<ref>\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?)?   # reference range (optional); may be wrapped in ( )
    \s*\)?
    \s*(?:[A-Za-z%/µμ\^\*0-9.]{1,20})?                # optional trailing unit repeated after the range
                                                       # (e.g. "12-16 g/dL") — common on real reports;
                                                       # not captured, just consumed so the line still matches
    \s*$
    """,
    re.VERBOSE,
)

DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})\b",
    re.IGNORECASE,
)


def offline_extractor(pages):
    """
    Deterministic, offline, rule-based extraction from raw page text.

    Supports both:
        Hemoglobin: 11.2 g/dL 12.0 - 16.0
    and:
        Hemoglobin: 11.2 g/dL
        Reference Range: 12.0 - 16.0 g/dL

    Reference ranges are ONLY taken from the report.
    """

    labs = []
    report_date = None

    # Matches a reference-range line separately.
    REFERENCE_LINE_PATTERN = re.compile(
        r"^\s*(?:reference\s*range|ref(?:erence)?\.?)\s*:"
        r"\s*(?P<range>\d+(?:\.\d+)?\s*[-–—]\s*\d+(?:\.\d+)?)"
        r"(?:\s*[A-Za-z%/µμ\^\*0-9.]*)?\s*$",
        re.IGNORECASE,
    )

    for page in pages:
        text = page.get("text") or ""
        page_number = page.get("page_number")

        if not report_date:
            m = DATE_PATTERN.search(text)
            if m:
                report_date = m.group(1)

        lines = [line.strip() for line in text.splitlines() if line.strip()]

        for line_index, line_stripped in enumerate(lines):

            # -----------------------------------------------------------
            # Reference range line:
            # Attach it to the immediately preceding lab result.
            # -----------------------------------------------------------
            ref_match = REFERENCE_LINE_PATTERN.match(line_stripped)

            if ref_match:
                if labs and labs[-1].get("page_number") == page_number:
                    ref_raw = ref_match.group("range").strip()
                    ref_low, ref_high = parse_reference_range(ref_raw)

                    labs[-1]["ref_low"] = ref_low
                    labs[-1]["ref_high"] = ref_high
                    labs[-1]["ref_raw"] = ref_raw

                    # Preserve both source lines as evidence.
                    previous_source = labs[-1].get("source_text") or ""
                    labs[-1]["source_text"] = (
                        previous_source + "\n" + line_stripped
                    ).strip()

                # Never create a fake lab called "Reference Range".
                continue

            # -----------------------------------------------------------
            # Ignore obvious metadata lines.
            # -----------------------------------------------------------
            if re.match(
                r"^(date|patient\s*name|name|age|sex|gender|"
                r"observations?|comments?|notes?)\s*:",
                line_stripped,
                re.IGNORECASE,
            ):
                continue

            # -----------------------------------------------------------
            # Normal lab line.
            # -----------------------------------------------------------
            m = LAB_LINE_PATTERN.match(line_stripped)

            if not m:
                continue

            name = m.group("name").strip()

            if len(name) < 2:
                continue

            # Prevent fake metadata/test names.
            if re.match(
                r"^(reference\s*range|date|patient\s*name|"
                r"name|age|sex|gender)$",
                name,
                re.IGNORECASE,
            ):
                continue

            value_str = m.group("value")
            unit = (m.group("unit") or "").strip() or None
            ref_raw = (m.group("ref") or "").strip() or None

            ref_low, ref_high = (
                parse_reference_range(ref_raw)
                if ref_raw
                else (None, None)
            )

            try:
                value = float(value_str)
            except (TypeError, ValueError):
                continue

            # -----------------------------------------------------------
            # Build the lab record.
            # -----------------------------------------------------------
            lab = {
                "test_name": name,
                "value": value,
                "unit": unit,
                "ref_low": ref_low,
                "ref_high": ref_high,
                "ref_raw": ref_raw,
                "page_number": page_number,
                "source_text": line_stripped,
                "confidence": 0.95,
            }

            labs.append(lab)

    # ---------------------------------------------------------------
    # Deduplicate identical extracted labs.
    # ---------------------------------------------------------------
    unique_labs = []
    seen = set()

    for lab in labs:
        key = (
            lab.get("test_name", "").strip().lower(),
            lab.get("value"),
            lab.get("unit"),
            lab.get("ref_low"),
            lab.get("ref_high"),
            lab.get("page_number"),
        )

        if key in seen:
            continue

        seen.add(key)
        unique_labs.append(lab)

    return {
        "labs": unique_labs,
        "report_date": report_date,
    }

def call_ai_extractor(pages):
    """
    Live-provider path. Calls the Anthropic API with a strict extraction
    prompt that mirrors the offline extractor's contract exactly (never
    invent values/units/ranges/dates/pages; null when unavailable).

    Only used when ANTHROPIC_API_KEY is configured server-side. On any
    failure, callers should fall back to offline_extractor().
    """
    import urllib.request

    full_text = "\n\n".join(f"[PAGE {p['page_number']}]\n{p['text']}" for p in pages)
    system_prompt = (
        "You extract structured lab data from medical report text. "
        "Rules: never invent values, units, reference ranges, dates or page "
        "numbers. If a field is not explicitly present in the text, return "
        "null for it. Preserve the exact source line as source_text. "
        "Respond ONLY with JSON matching this schema and nothing else: "
        '{"labs": [{"test_name": str, "value": number|null, "unit": string|null, '
        '"ref_low": number|null, "ref_high": number|null, "page_number": int, '
        '"source_text": string, "confidence": number}], "report_date": string|null}'
    )
    body = json.dumps({
        "model": "claude-sonnet-4-6",
        "max_tokens": 2000,
        "system": system_prompt,
        "messages": [{"role": "user", "content": full_text[:20000]}],
    }).encode("utf-8")

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=body,
        headers={
            "Content-Type": "application/json",
            "x-api-key": ANTHROPIC_API_KEY,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    text_out = "".join(
        block.get("text", "") for block in data.get("content", []) if block.get("type") == "text"
    )
    cleaned = re.sub(r"^```(json)?|```$", "", text_out.strip(), flags=re.MULTILINE).strip()
    parsed = json.loads(cleaned)

    labs = []
    for lab in parsed.get("labs", []):
        ref_low = lab.get("ref_low")
        ref_high = lab.get("ref_high")
        labs.append({
            "test_name": lab.get("test_name"),
            "value": lab.get("value"),
            "raw_value": str(lab.get("value")) if lab.get("value") is not None else None,
            "unit": lab.get("unit"),
            "ref_low": ref_low,
            "ref_high": ref_high,
            "ref_raw": f"{ref_low}-{ref_high}" if ref_low is not None and ref_high is not None else None,
            "page_number": lab.get("page_number"),
            "source_text": lab.get("source_text"),
            "confidence": lab.get("confidence", 0.7),
        })
    return {"labs": labs, "report_date": parsed.get("report_date")}


def run_extraction(pages):
    """Single entry point used by the API layer. Tries the live AI
    extractor if configured, and ALWAYS falls back to the deterministic
    offline extractor on any failure — extraction must never crash the
    application, and the app must work with zero API key."""
    if AI_ENABLED:
        try:
            return call_ai_extractor(pages), "AI_MODEL"
        except Exception as e:
            app.logger.warning("AI extraction failed, falling back to offline extractor: %s", e)
    return offline_extractor(pages), "OFFLINE_TEMPLATE"


# ---------------------------------------------------------------------------
# CONFLICT DETECTION — deterministic, never auto-resolved
# ---------------------------------------------------------------------------

CONFLICT_RULES = {
    "ALLERGY": {
        "no_known": lambda v: v.strip().lower() in {
            "no known allergies", "nka", "none", "no allergies", "nkda",
        },
    },
}


def detect_allergy_conflicts(db, patient_id, new_fact_row):
    """If a new allergy fact contradicts an existing 'no known allergies'
    fact (or vice versa), record a conflict requiring human verification."""
    new_value = new_fact_row["value"].strip()
    is_new_negative = CONFLICT_RULES["ALLERGY"]["no_known"](new_value)

    existing = db.execute(
        "SELECT * FROM patient_facts WHERE patient_id=? AND category='ALLERGY' "
        "AND id != ? AND verification_state != 'REJECTED' ORDER BY created_at ASC",
        (patient_id, new_fact_row["id"]),
    ).fetchall()

    conflicts_created = []
    for old in existing:
        old_value = old["value"].strip()
        is_old_negative = CONFLICT_RULES["ALLERGY"]["no_known"](old_value)
        contradictory = (is_new_negative != is_old_negative) or (
            not is_new_negative and not is_old_negative and old_value.lower() != new_value.lower()
        )
        if contradictory:
            conflict_id = new_id("conflict")
            db.execute(
                "INSERT INTO conflicts (id, patient_id, category, field_label, current_value, "
                "current_fact_id, previous_value, previous_fact_id, status, created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    conflict_id, patient_id, "ALLERGY", "Allergy",
                    new_value, new_fact_row["id"], old_value, old["id"],
                    "REQUIRES_HUMAN_VERIFICATION", now_iso(),
                ),
            )
            log_timeline(db, patient_id, "CONFLICT", now_iso(),
                         "Conflict detected: Allergy",
                         {"current": new_value, "previous": old_value})
            conflicts_created.append(conflict_id)
    return conflicts_created


# ---------------------------------------------------------------------------
# TIMELINE helper
# ---------------------------------------------------------------------------

def log_timeline(db, patient_id, event_type, event_date, title, detail: dict):
    db.execute(
        "INSERT INTO timeline_events (id, patient_id, event_type, event_date, title, detail, created_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (new_id("tl"), patient_id, event_type, event_date, title, json.dumps(detail), now_iso()),
    )


# ---------------------------------------------------------------------------
# AI SUMMARY — generated from the structured record, never raw PDF text
# ---------------------------------------------------------------------------

FORBIDDEN_SUMMARY_PATTERNS = [
    r"\bdiagnos",
    r"\bprescri",
    r"\brecommend(ed)? (a |an |the )?(dose|dosage|treatment|medication)",
    r"\bstart taking\b",
    r"\bstop taking\b",
    r"\bincrease your dose\b",
    r"\bdecrease your dose\b",
]


NEGATION_WINDOW_CHARS = 20


def safety_check_summary(text: str) -> str:
    """Defense-in-depth: flag any sentence that reads like a diagnosis,
    prescription, or dosage instruction, even if the summary template
    itself is otherwise safe. Never lets unsupported medical claims through.

    A match is ignored only when it sits directly after an explicit
    negation ("does not diagnose", "no prescription is given") — the
    disclaimer itself must be able to name the things it is refusing to
    do. Any non-negated occurrence still fails the check.
    """
    for pattern in FORBIDDEN_SUMMARY_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            window_start = max(0, m.start() - NEGATION_WINDOW_CHARS)
            preceding = text[window_start:m.start()].lower()
            if re.search(r"\b(not|no|never|without|n't)\b", preceding):
                continue  # negated — this is a safety disclaimer, not a claim
            raise ValueError(f"Summary failed safety validation (matched: {pattern})")
    return text


def generate_summary(db, patient_id) -> str:
    patient = db.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    if not patient:
        raise ValueError("Patient not found")

    labs = db.execute(
        "SELECT * FROM lab_results WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)
    ).fetchall()
    facts = db.execute(
        "SELECT * FROM patient_facts WHERE patient_id=? AND verification_state != 'REJECTED'",
        (patient_id,),
    ).fetchall()
    conflicts = db.execute(
        "SELECT * FROM conflicts WHERE patient_id=? AND status='REQUIRES_HUMAN_VERIFICATION'",
        (patient_id,),
    ).fetchall()

    lines = []
    age = patient["age"]
    sex = patient["sex"]
    if age or sex:
        lines.append(f"This record covers a {age or 'unknown age'}-year-old "
                      f"{sex or 'patient'} whose information has been organized from "
                      f"user-provided intake and uploaded reports.")

    abnormal = [l for l in labs if l["classification"] in ("LOW", "HIGH")]
    if abnormal:
        parts = [f"{l['test_name']} was reported as {l['classification']} "
                 f"({l['raw_value']} {l['unit'] or ''})".strip() for l in abnormal]
        lines.append("Reported lab values outside the source reference range: " + "; ".join(parts) + ".")
    elif labs:
        lines.append("All reported lab values fall within their source reference ranges.")

    allergies = [f["value"] for f in facts if f["category"] == "ALLERGY"]
    if allergies:
        lines.append("Reported allergy information: " + ", ".join(sorted(set(allergies))) + ".")

    unverified = [f for f in facts if f["verification_state"] == "UNVERIFIED"]
    if unverified:
        lines.append(f"{len(unverified)} extracted fact(s) remain unverified and should be reviewed.")

    if conflicts:
        lines.append(f"{len(conflicts)} conflict(s) between current and previous information "
                      f"require human verification before this record should be treated as reliable.")

    lines.append(
        "This is an organized summary of reported information only. It does not "
        "diagnose any condition, does not recommend treatment, and does not suggest "
        "medication or dosage changes. A qualified clinician should review this record."
    )

    text = " ".join(lines) if lines else (
        "No patient information or reports have been added yet. This is an organized "
        "summary of reported information only and does not provide diagnosis or treatment."
    )
    return safety_check_summary(text)


# ---------------------------------------------------------------------------
# Helpers to serialize the structured record
# ---------------------------------------------------------------------------

def json_or_empty(raw):
    try:
        return json.loads(raw) if raw else []
    except Exception:
        return []


def patient_to_dict(row):
    d = row_to_dict(row)
    if not d:
        return None
    for field in ("symptoms", "conditions", "allergies", "medications"):
        d[field] = json_or_empty(d.get(field))
    return d


def build_structured_record(db, patient_id):
    patient = db.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    if not patient:
        return None
    documents = rows_to_list(db.execute(
        "SELECT * FROM documents WHERE patient_id=? ORDER BY uploaded_at DESC", (patient_id,)
    ).fetchall())
    facts = rows_to_list(db.execute(
        "SELECT * FROM patient_facts WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)
    ).fetchall())
    labs = rows_to_list(db.execute(
        "SELECT * FROM lab_results WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)
    ).fetchall())
    evidence = rows_to_list(db.execute(
        "SELECT * FROM evidence WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)
    ).fetchall())
    conflicts = rows_to_list(db.execute(
        "SELECT * FROM conflicts WHERE patient_id=? ORDER BY created_at DESC", (patient_id,)
    ).fetchall())
    timeline = rows_to_list(db.execute(
        "SELECT * FROM timeline_events WHERE patient_id=? ORDER BY event_date DESC, created_at DESC",
        (patient_id,),
    ).fetchall())
    for t in timeline:
        t["detail"] = json_or_empty(t.get("detail"))

    return {
        "patient": patient_to_dict(patient),
        "documents": documents,
        "patient_facts": facts,
        "lab_results": labs,
        "evidence": evidence,
        "conflicts": conflicts,
        "timeline": timeline,
    }


# ---------------------------------------------------------------------------
# API ROUTES
# ---------------------------------------------------------------------------

def err(message, code=400):
    return jsonify({"error": message}), code


# ---- Global CORS / preflight handling --------------------------------------
# This runs before URL routing has a chance to 404 on an unmatched OPTIONS
# request, so every preflight from the browser gets a clean, CORS-headered
# response no matter which path it targets. This is the single most common
# reason a frontend can "see" the backend for simple GETs but fail on
# POST/PUT calls that carry a JSON body (those trigger a real preflight).
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        return "", 204


@app.after_request
def add_cors_headers(resp):
    requested_headers = request.headers.get("Access-Control-Request-Headers", "Content-Type")
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Headers"] = requested_headers
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    resp.headers["Access-Control-Max-Age"] = "86400"
    return resp


@app.route("/", methods=["GET"])
def root():
    # Serve the existing frontend/index.html unmodified, from the sibling
    # frontend/ directory, so the whole app is reachable from one process
    # at one host:port. index.html's own API_BASE already points at this
    # same origin, so no frontend change is needed.
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index_path):
        return send_from_directory(FRONTEND_DIR, "index.html")
    # Fallback (e.g. backend run standalone without the frontend/ folder
    # present) — keeps the previous reachability check working.
    return jsonify({
        "status": "ok",
        "service": "MedLens API",
        "hint": "If you can see this in your browser, the backend is reachable. "
                "Frontend calls should hit /api/... on this same host and port. "
                f"(frontend/index.html was not found at {index_path}.)",
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "ai_enabled": AI_ENABLED,
        "pdf_engine_available": PYPDF_AVAILABLE,
        "time": now_iso(),
    })


# ---- Patients -------------------------------------------------------------

@app.route("/api/patients", methods=["POST"])
def create_patient():
    body = request.get_json(force=True, silent=True) or {}
    db = get_db()
    pid = new_id("patient")
    ts = now_iso()
    symptoms = body.get("symptoms", [])
    conditions = body.get("conditions", [])
    allergies = body.get("allergies", [])
    medications = body.get("medications", [])

    db.execute(
        "INSERT INTO patients (id, age, sex, symptoms, conditions, allergies, medications, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, body.get("age"), body.get("sex"), json.dumps(symptoms), json.dumps(conditions),
         json.dumps(allergies), json.dumps(medications), ts, ts),
    )

    # Every user-supplied intake field becomes a provenance-tracked fact.
    for category, values in (
        ("SYMPTOM", symptoms), ("CONDITION", conditions),
        ("ALLERGY", allergies), ("MEDICATION", medications),
    ):
        for v in values:
            if not v:
                continue
            fact_id = new_id("fact")
            db.execute(
                "INSERT INTO patient_facts (id, patient_id, category, value, origin, verification_state, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (fact_id, pid, category, v, "USER_PROVIDED", "VERIFIED", ts, ts),
            )
            if category == "ALLERGY":
                new_fact = db.execute("SELECT * FROM patient_facts WHERE id=?", (fact_id,)).fetchone()
                detect_allergy_conflicts(db, pid, new_fact)

    log_timeline(db, pid, "PATIENT_FACT", ts, "Patient intake recorded",
                 {"age": body.get("age"), "sex": body.get("sex")})
    db.commit()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    return jsonify(patient_to_dict(patient)), 201


@app.route("/api/patients/<pid>", methods=["GET"])
def get_patient(pid):
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not patient:
        return err("Patient not found", 404)
    return jsonify(patient_to_dict(patient))


@app.route("/api/patients/<pid>", methods=["PUT"])
def update_patient(pid):
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not patient:
        return err("Patient not found", 404)
    body = request.get_json(force=True, silent=True) or {}
    ts = now_iso()

    age = body.get("age", patient["age"])
    sex = body.get("sex", patient["sex"])
    symptoms = body.get("symptoms", json_or_empty(patient["symptoms"]))
    conditions = body.get("conditions", json_or_empty(patient["conditions"]))
    allergies = body.get("allergies", json_or_empty(patient["allergies"]))
    medications = body.get("medications", json_or_empty(patient["medications"]))

    db.execute(
        "UPDATE patients SET age=?, sex=?, symptoms=?, conditions=?, allergies=?, medications=?, updated_at=? "
        "WHERE id=?",
        (age, sex, json.dumps(symptoms), json.dumps(conditions), json.dumps(allergies),
         json.dumps(medications), ts, pid),
    )

    # New allergy values entered via edit are tracked + conflict-checked too.
    existing_allergy_values = {
        r["value"] for r in db.execute(
            "SELECT value FROM patient_facts WHERE patient_id=? AND category='ALLERGY'", (pid,)
        ).fetchall()
    }
    for v in allergies:
        if v and v not in existing_allergy_values:
            fact_id = new_id("fact")
            db.execute(
                "INSERT INTO patient_facts (id, patient_id, category, value, origin, verification_state, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (fact_id, pid, "ALLERGY", v, "USER_PROVIDED", "VERIFIED", ts, ts),
            )
            new_fact = db.execute("SELECT * FROM patient_facts WHERE id=?", (fact_id,)).fetchone()
            detect_allergy_conflicts(db, pid, new_fact)

    db.commit()
    updated = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    return jsonify(patient_to_dict(updated))


@app.route("/api/patients/<pid>/record", methods=["GET"])
def get_record(pid):
    db = get_db()
    record = build_structured_record(db, pid)
    if record is None:
        return err("Patient not found", 404)
    return jsonify(record)


@app.route("/api/patients/<pid>/labs", methods=["GET"])
def get_labs(pid):
    db = get_db()
    labs = rows_to_list(db.execute(
        "SELECT * FROM lab_results WHERE patient_id=? ORDER BY created_at DESC", (pid,)
    ).fetchall())
    return jsonify(labs)


@app.route("/api/patients/<pid>/evidence", methods=["GET"])
def get_evidence(pid):
    db = get_db()
    evidence = rows_to_list(db.execute(
        "SELECT * FROM evidence WHERE patient_id=? ORDER BY created_at DESC", (pid,)
    ).fetchall())
    return jsonify(evidence)


@app.route("/api/patients/<pid>/conflicts", methods=["GET"])
def get_conflicts(pid):
    db = get_db()
    conflicts = rows_to_list(db.execute(
        "SELECT * FROM conflicts WHERE patient_id=? ORDER BY created_at DESC", (pid,)
    ).fetchall())
    return jsonify(conflicts)


@app.route("/api/patients/<pid>/timeline", methods=["GET"])
def get_timeline(pid):
    db = get_db()
    timeline = rows_to_list(db.execute(
        "SELECT * FROM timeline_events WHERE patient_id=? ORDER BY event_date DESC, created_at DESC",
        (pid,),
    ).fetchall())
    for t in timeline:
        t["detail"] = json_or_empty(t.get("detail"))
    return jsonify(timeline)


# ---- Reports / documents ---------------------------------------------------

@app.route("/api/reports/upload", methods=["POST"])
def upload_report():
    patient_id = request.form.get("patient_id") or (request.get_json(silent=True) or {}).get("patient_id")
    if not patient_id:
        return err("patient_id is required")
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
    if not patient:
        return err("Patient not found", 404)

    if "file" not in request.files:
        return err("No file part in request")
    f = request.files["file"]
    if f.filename == "":
        return err("No file selected")
    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return err("Only PDF files are supported", 415)

    filename = secure_filename(f.filename) or f"report{ext}"
    doc_id = new_id("doc")
    stored_name = f"{doc_id}{ext}"
    stored_path = os.path.join(UPLOAD_DIR, stored_name)
    file_bytes = f.read()
    if len(file_bytes) == 0:
        return err("Uploaded file is empty")
    with open(stored_path, "wb") as out:
        out.write(file_bytes)

    ts = now_iso()
    db.execute(
        "INSERT INTO documents (id, patient_id, filename, stored_path, status, uploaded_at) "
        "VALUES (?,?,?,?,?,?)",
        (doc_id, patient_id, filename, stored_path, "UPLOADED", ts),
    )
    log_timeline(db, patient_id, "REPORT_UPLOADED", ts, f"Report uploaded: {filename}", {"document_id": doc_id})
    db.commit()

    doc = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return jsonify(row_to_dict(doc)), 201
@app.route("/api/patients/<pid>/reports", methods=["GET"])
def get_patient_reports(pid):
    db = get_db()

    patient = db.execute(
        "SELECT id FROM patients WHERE id=?",
        (pid,)
    ).fetchone()

    if not patient:
        return err("Patient not found", 404)

    reports = rows_to_list(db.execute(
        """
        SELECT id, patient_id, filename, report_date, status,
               error_message, uploaded_at
        FROM documents
        WHERE patient_id=?
        ORDER BY uploaded_at DESC
        """,
        (pid,)
    ).fetchall())

    return jsonify(reports)

@app.route("/api/reports/<doc_id>", methods=["GET"])
def get_report(doc_id):
    db = get_db()
    doc = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        return err("Document not found", 404)
    pages = rows_to_list(db.execute(
        "SELECT page_number, text FROM document_pages WHERE document_id=? ORDER BY page_number",
        (doc_id,),
    ).fetchall())
    result = row_to_dict(doc)
    result["pages"] = pages
    return jsonify(result)


@app.route("/api/reports/<doc_id>/extract", methods=["POST"])
def extract_report(doc_id):
    db = get_db()
    doc = db.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    if not doc:
        return err("Document not found", 404)

    try:
        with open(doc["stored_path"], "rb") as f:
            file_bytes = f.read()
        pages, had_text = extract_pdf_pages(file_bytes)
    except PdfExtractionError as e:
        db.execute("UPDATE documents SET status=?, error_message=? WHERE id=?",
                   ("EXTRACTION_FAILED", str(e), doc_id))
        db.commit()
        return err(f"Extraction failed: {e}", 422)
    except Exception as e:
        db.execute("UPDATE documents SET status=?, error_message=? WHERE id=?",
                   ("EXTRACTION_FAILED", "Unexpected error during extraction", doc_id))
        db.commit()
        app.logger.exception("Unexpected extraction error")
        return err("Unexpected error during extraction", 500)

    ts = now_iso()
    for p in pages:
        db.execute(
            "INSERT INTO document_pages (id, document_id, page_number, text) VALUES (?,?,?,?)",
            (new_id("page"), doc_id, p["page_number"], p["text"]),
        )

    extraction, extraction_method = run_extraction(pages)
    patient_id = doc["patient_id"]
    report_date = extraction.get("report_date")

    created_labs = []
    for lab in extraction["labs"]:
        classification = classify_value(lab.get("value"), lab.get("ref_low"), lab.get("ref_high"))
        lab_id = new_id("lab")
        db.execute(
            "INSERT INTO lab_results (id, patient_id, document_id, test_name, value, raw_value, unit, "
            "ref_low, ref_high, ref_raw, classification, report_date, page_number, source_text, "
            "confidence, origin, verification_state, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                lab_id, patient_id, doc_id, lab.get("test_name"), lab.get("value"), lab.get("raw_value"),
                lab.get("unit"), lab.get("ref_low"), lab.get("ref_high"), lab.get("ref_raw"),
                classification, report_date, lab.get("page_number"), lab.get("source_text"),
                lab.get("confidence"), "AI_GENERATED" if extraction_method == "AI_MODEL" else "REPORT_EXTRACTED",
                "UNVERIFIED", ts,
            ),
        )
        db.execute(
            "INSERT INTO evidence (id, patient_id, subject_type, subject_id, document_id, page_number, "
            "source_text, method, confidence, origin, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (
                new_id("ev"), patient_id, "LAB_RESULT", lab_id, doc_id, lab.get("page_number"),
                lab.get("source_text"), "Deterministic reference-range comparison",
                lab.get("confidence"), "REPORT_EXTRACTED", ts,
            ),
        )
        log_timeline(db, patient_id, "LAB_RESULT", report_date or ts,
                     f"{lab.get('test_name')}: {lab.get('raw_value')} {lab.get('unit') or ''} -> {classification}".strip(),
                     {"lab_id": lab_id, "classification": classification})
        created_labs.append(lab_id)

    db.execute(
        "UPDATE documents SET status=?, report_date=? WHERE id=?",
        ("EXTRACTED", report_date, doc_id),
    )
    db.commit()

    return jsonify({
        "document_id": doc_id,
        "status": "EXTRACTED",
        "pages_processed": len(pages),
        "had_selectable_text": had_text,
        "extraction_method": extraction_method,
        "labs_created": created_labs,
        "report_date": report_date,
    })


# ---- Facts / verification --------------------------------------------------

@app.route("/api/facts/<fact_id>/verify", methods=["POST"])
def verify_fact(fact_id):
    db = get_db()
    fact = db.execute("SELECT * FROM patient_facts WHERE id=?", (fact_id,)).fetchone()
    is_lab = False
    if not fact:
        fact = db.execute("SELECT * FROM lab_results WHERE id=?", (fact_id,)).fetchone()
        is_lab = True
    if not fact:
        return err("Fact not found", 404)

    body = request.get_json(force=True, silent=True) or {}
    action = (body.get("action") or "VERIFY").upper()
    if action not in ("VERIFY", "REJECT"):
        return err("action must be VERIFY or REJECT")

    ts = now_iso()
    table = "lab_results" if is_lab else "patient_facts"
    new_state = "VERIFIED" if action == "VERIFY" else "REJECTED"
    db.execute(f"UPDATE {table} SET verification_state=? WHERE id=?", (new_state, fact_id))
    if not is_lab:
        db.execute("UPDATE patient_facts SET updated_at=? WHERE id=?", (ts, fact_id))

    db.execute(
        "INSERT INTO verification_events (id, patient_id, subject_type, subject_id, action, previous_value, "
        "new_value, note, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            new_id("vev"), fact["patient_id"], "LAB_RESULT" if is_lab else "PATIENT_FACT", fact_id,
            action, fact["verification_state"], new_state, body.get("note"), ts,
        ),
    )
    log_timeline(db, fact["patient_id"], "VERIFICATION", ts,
                 f"Fact {new_state.lower()}: {fact['test_name'] if is_lab else fact['value']}",
                 {"fact_id": fact_id, "new_state": new_state})
    db.commit()

    updated = db.execute(f"SELECT * FROM {table} WHERE id=?", (fact_id,)).fetchone()
    return jsonify(row_to_dict(updated))


@app.route("/api/facts/<fact_id>", methods=["PUT"])
def edit_fact(fact_id):
    db = get_db()
    fact = db.execute("SELECT * FROM patient_facts WHERE id=?", (fact_id,)).fetchone()
    is_lab = False
    if not fact:
        fact = db.execute("SELECT * FROM lab_results WHERE id=?", (fact_id,)).fetchone()
        is_lab = True
    if not fact:
        return err("Fact not found", 404)

    body = request.get_json(force=True, silent=True) or {}
    ts = now_iso()
    previous_value = fact["value"] if not is_lab else fact["raw_value"]

    if is_lab:
        new_value = body.get("value", fact["value"])
        new_ref_low = body.get("ref_low", fact["ref_low"])
        new_ref_high = body.get("ref_high", fact["ref_high"])
        classification = classify_value(new_value, new_ref_low, new_ref_high)
        db.execute(
            "UPDATE lab_results SET value=?, raw_value=?, unit=?, ref_low=?, ref_high=?, classification=?, "
            "verification_state='VERIFIED' WHERE id=?",
            (new_value, str(new_value), body.get("unit", fact["unit"]), new_ref_low, new_ref_high,
             classification, fact_id),
        )
        new_value_repr = str(new_value)
    else:
        new_value = body.get("value", fact["value"])
        db.execute(
            "UPDATE patient_facts SET value=?, verification_state='VERIFIED', updated_at=? WHERE id=?",
            (new_value, ts, fact_id),
        )
        new_value_repr = new_value

    db.execute(
        "INSERT INTO verification_events (id, patient_id, subject_type, subject_id, action, previous_value, "
        "new_value, note, created_at) VALUES (?,?,?,?,?,?,?,?,?)",
        (
            new_id("vev"), fact["patient_id"], "LAB_RESULT" if is_lab else "PATIENT_FACT", fact_id,
            "EDIT", previous_value, new_value_repr, body.get("note"), ts,
        ),
    )
    log_timeline(db, fact["patient_id"], "VERIFICATION", ts, f"Fact edited",
                 {"fact_id": fact_id, "previous": previous_value, "new": new_value_repr})
    db.commit()

    table = "lab_results" if is_lab else "patient_facts"
    updated = db.execute(f"SELECT * FROM {table} WHERE id=?", (fact_id,)).fetchone()
    return jsonify(row_to_dict(updated))


# ---- Summary ---------------------------------------------------------------

@app.route("/api/patients/<pid>/summary", methods=["POST"])
def create_summary(pid):
    db = get_db()
    patient = db.execute("SELECT * FROM patients WHERE id=?", (pid,)).fetchone()
    if not patient:
        return err("Patient not found", 404)
    try:
        text = generate_summary(db, pid)
    except ValueError as e:
        return err(str(e), 422)

    ts = now_iso()
    summary_id = new_id("summary")
    generated_by = "AI_MODEL" if AI_ENABLED else "OFFLINE_TEMPLATE"
    db.execute(
        "INSERT INTO summaries (id, patient_id, text, generated_by, created_at) VALUES (?,?,?,?,?)",
        (summary_id, pid, text, generated_by, ts),
    )
    db.commit()
    return jsonify({"id": summary_id, "patient_id": pid, "text": text, "generated_by": generated_by,
                     "created_at": ts, "label": "AI-GENERATED SUMMARY"})


# ---- Demo -------------------------------------------------------------------

@app.route("/api/demo/patient", methods=["GET"])
def demo_patient():
    """Builds a full synthetic demo patient end-to-end (intake, previous +
    current reports, extraction, classification, conflict, timeline,
    summary) and returns the new patient id plus the full structured record.
    Idempotent per call — always creates a fresh demo patient so repeated
    demo clicks don't collide with a prior run."""
    db = get_db()
    ts = now_iso()
    pid = new_id("patient")

    db.execute(
        "INSERT INTO patients (id, age, sex, symptoms, conditions, allergies, medications, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?)",
        (pid, 41, "female", json.dumps(["fatigue", "mild dizziness"]), json.dumps(["hypertension"]),
         json.dumps(["Penicillin"]), json.dumps(["Lisinopril 10mg"]), ts, ts),
    )
    for category, values in (
        ("SYMPTOM", ["fatigue", "mild dizziness"]), ("CONDITION", ["hypertension"]),
        ("ALLERGY", ["Penicillin"]), ("MEDICATION", ["Lisinopril 10mg"]),
    ):
        for v in values:
            fact_id = new_id("fact")
            db.execute(
                "INSERT INTO patient_facts (id, patient_id, category, value, origin, verification_state, "
                "created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (fact_id, pid, category, v, "USER_PROVIDED", "VERIFIED", ts, ts),
            )
    log_timeline(db, pid, "PATIENT_FACT", ts, "Demo patient intake recorded", {"age": 41, "sex": "female"})

    # --- Previous report (document, no PDF file, synthetic) ---
    prev_doc_id = new_id("doc")
    prev_date = (datetime.date.today() - datetime.timedelta(days=90)).isoformat()
    db.execute(
        "INSERT INTO documents (id, patient_id, filename, stored_path, report_date, status, uploaded_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (prev_doc_id, pid, "CBC_Report_Previous.pdf (synthetic demo)", None, prev_date, "EXTRACTED", ts),
    )
    prev_hgb_source = "Hemoglobin 13.1 g/dL 12–16 g/dL"
    prev_hgb_id = new_id("lab")
    prev_class = classify_value(13.1, 12, 16)
    db.execute(
        "INSERT INTO lab_results (id, patient_id, document_id, test_name, value, raw_value, unit, ref_low, "
        "ref_high, ref_raw, classification, report_date, page_number, source_text, confidence, origin, "
        "verification_state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (prev_hgb_id, pid, prev_doc_id, "Hemoglobin", 13.1, "13.1", "g/dL", 12, 16, "12–16", prev_class,
         prev_date, 1, prev_hgb_source, 0.95, "REPORT_EXTRACTED", "VERIFIED", ts),
    )
    db.execute(
        "INSERT INTO evidence (id, patient_id, subject_type, subject_id, document_id, page_number, "
        "source_text, method, confidence, origin, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
        (new_id("ev"), pid, "LAB_RESULT", prev_hgb_id, prev_doc_id, 1, prev_hgb_source,
         "Deterministic reference-range comparison", 0.95, "REPORT_EXTRACTED", ts),
    )
    prev_allergy_fact_id = new_id("fact")
    db.execute(
        "INSERT INTO patient_facts (id, patient_id, category, value, origin, verification_state, "
        "document_id, page_number, source_text, confidence, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (prev_allergy_fact_id, pid, "ALLERGY", "No known allergies", "REPORT_EXTRACTED", "VERIFIED",
         prev_doc_id, 1, "Allergies: No known allergies (NKDA)", 0.9, ts, ts),
    )
    log_timeline(db, pid, "REPORT_UPLOADED", prev_date, "Previous report: CBC_Report_Previous.pdf",
                 {"document_id": prev_doc_id})
    log_timeline(db, pid, "LAB_RESULT", prev_date, f"Hemoglobin: 13.1 g/dL -> {prev_class}",
                 {"lab_id": prev_hgb_id, "classification": prev_class})

    # --- Current report ---
    cur_doc_id = new_id("doc")
    cur_date = datetime.date.today().isoformat()
    db.execute(
        "INSERT INTO documents (id, patient_id, filename, stored_path, report_date, status, uploaded_at) "
        "VALUES (?,?,?,?,?,?,?)",
        (cur_doc_id, pid, "CBC_Report.pdf (synthetic demo)", None, cur_date, "EXTRACTED", ts),
    )
    demo_labs = [
        ("Hemoglobin", 11.2, "g/dL", 12, 16, "Hemoglobin 11.2 g/dL 12–16 g/dL", 2),
        ("Glucose", 94, "mg/dL", 70, 100, "Glucose 94 mg/dL 70–100 mg/dL", 2),
    ]
    current_hgb_id = None
    for name, value, unit, lo, hi, source_text, page in demo_labs:
        lab_id = new_id("lab")
        classification = classify_value(value, lo, hi)
        if name == "Hemoglobin":
            current_hgb_id = lab_id
        db.execute(
            "INSERT INTO lab_results (id, patient_id, document_id, test_name, value, raw_value, unit, ref_low, "
            "ref_high, ref_raw, classification, report_date, page_number, source_text, confidence, origin, "
            "verification_state, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (lab_id, pid, cur_doc_id, name, value, str(value), unit, lo, hi, f"{lo}–{hi}", classification,
             cur_date, page, source_text, 0.95, "REPORT_EXTRACTED", "UNVERIFIED", ts),
        )
        db.execute(
            "INSERT INTO evidence (id, patient_id, subject_type, subject_id, document_id, page_number, "
            "source_text, method, confidence, origin, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (new_id("ev"), pid, "LAB_RESULT", lab_id, cur_doc_id, page, source_text,
             "Deterministic reference-range comparison", 0.95, "REPORT_EXTRACTED", ts),
        )
        log_timeline(db, pid, "LAB_RESULT", cur_date, f"{name}: {value} {unit} -> {classification}",
                     {"lab_id": lab_id, "classification": classification,
                      "previous_value": 13.1 if name == "Hemoglobin" else None,
                      "numeric_change": round(value - 13.1, 2) if name == "Hemoglobin" else None})
    log_timeline(db, pid, "REPORT_UPLOADED", cur_date, "Current report: CBC_Report.pdf",
                 {"document_id": cur_doc_id})

    # --- Current intake introduces the allergy conflict ---
    cur_allergy_fact = db.execute(
        "SELECT * FROM patient_facts WHERE patient_id=? AND category='ALLERGY' AND value='Penicillin'",
        (pid,),
    ).fetchone()
    detect_allergy_conflicts(db, pid, cur_allergy_fact)

    db.commit()

    try:
        summary_text = generate_summary(db, pid)
    except ValueError:
        summary_text = None
    if summary_text:
        db.execute(
            "INSERT INTO summaries (id, patient_id, text, generated_by, created_at) VALUES (?,?,?,?,?)",
            (new_id("summary"), pid, summary_text,
             "AI_MODEL" if AI_ENABLED else "OFFLINE_TEMPLATE", now_iso()),
        )
        db.commit()

    record = build_structured_record(db, pid)
    record["patient_id"] = pid
    return jsonify(record)


# ---------------------------------------------------------------------------
# Error handlers — never leak stack traces
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "Uploaded file exceeds the maximum allowed size"}), 413


@app.errorhandler(500)
def server_error(e):
    app.logger.exception("Internal server error")
    return jsonify({"error": "Internal server error"}), 500


def _open_browser_when_ready(host, port, path="/", timeout_seconds=15):
    """Poll the port until Flask is actually accepting connections, then
    open the default browser. Runs on a background thread so it never
    blocks/delays app.run(); any failure (headless box, no browser, etc.)
    is swallowed so it can never crash or stall the server itself."""
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.5):
                break
        except OSError:
            time.sleep(0.15)
    else:
        return  # server never came up within timeout — just skip opening
    try:
        webbrowser.open(f"http://{host}:{port}{path}")
    except Exception:
        pass  # e.g. no GUI browser available — non-fatal


if __name__ == "__main__":
    init_db()
    port = int(os.environ.get("PORT", "8000"))
    debug_mode = os.environ.get("DEBUG", "0") == "1"

    # Auto-open the browser once the server is actually ready to accept
    # connections. Gated so it fires exactly once even under Flask's
    # debug reloader (which re-executes this script in a child process
    # with WERKZEUG_RUN_MAIN=true) and can be disabled for automated
    # testing with MEDLENS_OPEN_BROWSER=0.
    open_browser = os.environ.get("MEDLENS_OPEN_BROWSER", "1") != "0"
    is_reloader_child = os.environ.get("WERKZEUG_RUN_MAIN") == "true"
    if open_browser and (not debug_mode or is_reloader_child):
        threading.Thread(
            target=_open_browser_when_ready,
            args=("127.0.0.1", port),
            daemon=True,
        ).start()

    app.run(host="0.0.0.0", port=port, threaded=True, debug=debug_mode)
else:
    init_db()
