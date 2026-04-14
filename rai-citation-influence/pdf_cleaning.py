import re
import pdfplumber

# ===================== PDF LOADING (pdfplumber) =========================

def load_pdf_text(path: str) -> str:
    text = ""
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
            except Exception as e:
                print(f"[WARN] Failed to extract page {i} in {path}: {e}")
    return text

# ===================== CLEANING =========================

def clean_text(text: str) -> str:
    # ---------------- LIGATURE FIXES ----------------
    ligatures = {
        "ﬁ": "fi",
        "ﬂ": "fl",
        "ﬀ": "ff",
        "ﬃ": "ffi",
        "ﬄ": "ffl",
    }
    for lig, repl in ligatures.items():
        text = text.replace(lig, repl)

    # ---------------- FIX BROKEN WORDS ----------------
    text = re.sub(r'\b([A-Z]{2,})\s+([A-Z]{2,})\b', r'\1\2', text)

    # Fix hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # ---------------- REMOVE HEADERS / FOOTERS ----------------
    text = re.sub(r'\b(Page|PAGE)\s*\d+(\s*of\s*\d+)?\b', ' ', text)
    text = re.sub(r'\b\d{1,2},?\s+\d{4}\b', ' ', text)

    # ---------------- SMART NUMBER HANDLING ----------------
    # Keep numbers if tied to structural words
    text = re.sub(
        r'(?<!section )(?<!article )(?<!topic )(?<!chapter )(?<!part )\b\d+\b',
        ' ',
        text,
        flags=re.IGNORECASE
    )

    # ---------------- PRESERVE STRUCTURE ----------------
    text = re.sub(r'\n{2,}', '\n\n', text)  # preserve paragraphs
    text = re.sub(r'[ \t]+', ' ', text)      # normalize spaces

    # ---------------- REMOVE ARTIFACT CHARACTERS ----------------
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[_\-]{2,}', ' ', text)

    return text.strip()
