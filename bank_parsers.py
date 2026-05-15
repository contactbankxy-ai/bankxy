"""
=============================================================================
BANKXY — Bank Parsers (Refactored)
=============================================================================
Architecture:
  BaseParser  (abstract)
    ├─ HDFCParser
    │    ├─ HDFCParser_v1  (lines-based)
    │    └─ HDFCParser_v2  (text-based fallback)
    ├─ UnionBankParser
    ├─ CanaraParser
    ├─ GenericParser
    └─ ParserFactory  (auto-detect + fallback chain)

Each parser returns:
  (pandas.DataFrame, page_count: int, parser_name: str)
so the ValidationEngine has everything it needs.
=============================================================================
"""

import pdfplumber
import pandas as pd
import re
import logging
from abc import ABC, abstractmethod
from typing import Optional

logger = logging.getLogger("bankxy.parsers")


# ===========================================================================
# Shared numeric cleaner
# ===========================================================================
def clean_numeric(value) -> object:
    if isinstance(value, str):
        s = value.replace('\n', '').replace(' ', '').replace(',', '').strip()
        match = re.match(r'^-?\d*\.?\d+$', s)
        if match:
            try:
                return float(s)
            except ValueError:
                pass
    return value


def clean_currency(val) -> Optional[float]:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except (TypeError, ValueError):
        pass
    s = str(val).replace(',', '').replace(' ', '').strip()
    s = re.sub(r'[^\d.\-]', '', s)
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


# ===========================================================================
# Base Parser
# ===========================================================================
class BaseParser(ABC):
    name: str = "BaseParser"

    @abstractmethod
    def parse(self, pdf_path: str, password: Optional[str] = None) -> pd.DataFrame:
        """Parse the PDF and return a raw, uncleaned DataFrame."""
        ...

    def run(self, pdf_path: str, password: Optional[str] = None) -> tuple:
        """
        Entry point.  Returns (DataFrame, page_count, parser_name).
        Raises ValueError on failure.
        """
        with pdfplumber.open(pdf_path, password=password) as pdf:
            page_count = len(pdf.pages)

        logger.info("Running parser: %s | pages=%d", self.name, page_count)
        df = self.parse(pdf_path, password)
        return df, page_count, self.name


# ===========================================================================
# HDFC Parser v1 — lines-based table extraction (preferred)
# ===========================================================================
class HDFCParser_v1(BaseParser):
    """
    Text-line-based HDFC parser.

    Why NOT extract_table():
      pdfplumber's table extractor stacks all transactions on a page into ONE
      row, with each column holding a different number of \\n-separated values.
      The narration column wraps to 2-3 lines per transaction while date/ref/
      balance always have exactly 1 line per transaction.  Any positional-index
      approach on these mismatched columns is fundamentally broken.

    Why extract_text() is correct:
      The raw text renders each transaction on its own physical line, exactly as
      printed on paper.  Multi-line narrations appear as continuation lines with
      no date at the start.  This mirrors the document's actual structure and
      gives us the exact amount (withdrawal OR deposit) directly from the line —
      no balance-diff guessing needed.

    Transaction line format:
      DD/MM/YY  <narration>  <ref_no>  DD/MM/YY  [amount]  closing_balance
    """
    name = "HDFCParser_v1"

    # Lines that start a new transaction
    _TXN_RE = re.compile(
        r"^(\d{2}/\d{2}/\d{2})"            # Date
        r"\s+(.+?)"                          # Narration (non-greedy)
        r"\s+([\w\s-]{5,25}?)"               # Ref No (alphanumeric, 5-25 chars)
        r"\s+(\d{2}/\d{2}/\d{2})"           # Value Date
        r"(?:\s+(-?[\d,]+\.\d{2}))?"          # optional Amount1 (can be negative)
        r"(?:\s+(-?[\d,]+\.\d{2}))?"          # optional Amount2 (can be negative)
        r"\s+(-?[\d,]+\.\d{2})\s*$"           # Closing Balance (always last)
    )

    # Lines to skip — page headers, footers, address blocks, etc.
    _SKIP_RE = re.compile(
        r"(HDFC\s*BANK"
        r"|Closing\s*balance\s*includes"
        r"|Contents\s*of\s*this\s*statement"
        r"|State\s*account\s*branch"
        r"|RegisteredOffice"
        r"|thisstatement\."
        r"|Page\s*No\.?:"
        r"|StatementFrom\s*:"
        r"|Date\s+Narration\s+Chq"
        r"|AccountBranch\s*:"
        r"|AccountNo\s*:"
        r"|AccountType\s*:"
        r"|AccountStatus\s*:"
        r"|CustID\s*:"
        r"|Nomination\s*:"
        r"|Address\s*:"
        r"|BranchCode\s*:"
        r"|JointHolder"
        r"|RTGS/NEFT"
        r"|MICR:"
        r"|Currency:INR"
        r"|ODLimit\s*:"
        r"|A/COpenDate"
        r"|Phoneno\."
        r"|Email\s*:"
        r"|City\s*:"
        r"|State\s*:)"
        , re.IGNORECASE
    )

    @staticmethod
    def _clean(s) -> float | None:
        if not s:
            return None
        s = s.replace(",", "").strip()
        try:
            return float(s)
        except ValueError:
            return None

    def parse(self, pdf_path: str, password=None) -> pd.DataFrame:
        rows     = []
        current  = None
        prev_bal = None

        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""

                for raw_line in text.split("\n"):
                    line = raw_line.strip()
                    if not line or self._SKIP_RE.match(line):
                        current = None  # reset on page-boundary junk lines
                        continue

                    m = self._TXN_RE.match(line)
                    if m:
                        date, narr, ref, vdt, a1, a2, bal_str = m.groups()
                        balc = self._clean(bal_str)
                        a1c  = self._clean(a1)
                        a2c  = self._clean(a2)

                        # Determine Withdrawal / Deposit
                        # Two amounts on one line (rare): first=W, second=D
                        if a1c is not None and a2c is not None:
                            withdrawal, deposit = a1c, a2c
                        elif a1c is not None:
                            # Single amount — use mathematical difference to classify
                            if prev_bal is not None and balc is not None:
                                diff = round(balc - prev_bal, 2)
                                # A negative withdrawal is mathematically a deposit to the balance
                                # So we must check if diff matches a1c or -a1c
                                if abs(diff - a1c) < 0.01:
                                    withdrawal, deposit = None, a1c
                                elif abs(diff - (-a1c)) < 0.01:
                                    withdrawal, deposit = a1c, None
                                else:
                                    # Fallback if there's an unexplained gap
                                    if diff < 0:
                                        withdrawal, deposit = a1c, None
                                    else:
                                        withdrawal, deposit = None, a1c
                            else:
                                # Very first transaction, opening balance unknown
                                # Use word coordinates to classify the single amount
                                withdrawal, deposit = None, None
                                words = page.extract_words()
                                for w in words:
                                    if w["text"] == a1:
                                        if w["x0"] < 480:
                                            withdrawal = a1c
                                        else:
                                            deposit = a1c
                                        break
                        else:
                            withdrawal, deposit = None, None

                        current = {
                            "Date":             date,
                            "Narration":        narr.strip(),
                            "Chq./Ref.No.":     ref.strip(),
                            "Value Dt":         vdt,
                            "Withdrawal Amt.":  withdrawal,
                            "Deposit Amt.":     deposit,
                            "Closing Balance":  balc,
                        }
                        rows.append(current)
                        prev_bal = balc

                    elif current is not None and line:
                        # Continuation: append to narration of current transaction
                        current["Narration"] = (current["Narration"] + " " + line).strip()

        if not rows:
            raise ValueError("HDFCParser_v1: No transaction lines found in PDF text.")

        df = pd.DataFrame(rows)
        # Final type enforcement
        df["Withdrawal Amt."]  = pd.to_numeric(df["Withdrawal Amt."],  errors="coerce")
        df["Deposit Amt."]     = pd.to_numeric(df["Deposit Amt."],     errors="coerce")
        df["Closing Balance"]  = pd.to_numeric(df["Closing Balance"],  errors="coerce")
        df = df[df["Date"].str.match(r"\d{2}/\d{2}/\d{2}", na=False)]
        df.reset_index(drop=True, inplace=True)
        return df


# HDFCParser_v2 kept as a table-based fallback for alternate HDFC layouts
# that do NOT use the same text format (e.g., branch-generated PDFs)
class UnionBankParser(BaseParser):
    name = "UnionBankParser"

    def parse(self, pdf_path: str, password: Optional[str] = None) -> pd.DataFrame:
        all_rows = []
        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                for row in table:
                    if any(row):
                        all_rows.append(row)

        if not all_rows:
            raise ValueError("UnionBankParser: No table found.")

        columns = ['S.No', 'Date', 'Transaction Id', 'Remarks', 'Amount(Rs.)', 'Balance(Rs.)']
        df = pd.DataFrame(all_rows[1:], columns=columns)
        df = df[df['S.No'] != 'S.No']
        df.reset_index(drop=True, inplace=True)

        def split_amount(value):
            try:
                if pd.isna(value):
                    return pd.Series([None, None])
            except (TypeError, ValueError):
                pass
            amount_str = str(value).strip()
            if '(Dr)' in amount_str:
                return pd.Series([float(amount_str.replace('(Dr)', '').replace(',', '').strip()), None])
            elif '(Cr)' in amount_str:
                return pd.Series([None, float(amount_str.replace('(Cr)', '').replace(',', '').strip())])
            return pd.Series([None, None])

        df[['DR', 'CR']] = df['Amount(Rs.)'].apply(split_amount)
        df = df[pd.to_numeric(df['S.No'], errors='coerce').notnull()]
        df.reset_index(drop=True, inplace=True)
        return df


# ===========================================================================
# Canara Bank Parser
# ===========================================================================
class CanaraParser(BaseParser):
    name = "CanaraParser"

    def parse(self, pdf_path: str, password: Optional[str] = None) -> pd.DataFrame:
        all_rows = []
        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                for row in table:
                    if any(row):
                        all_rows.append(row)

        if not all_rows:
            raise ValueError("CanaraParser: No table found.")

        columns = all_rows[0]
        df = pd.DataFrame(all_rows[1:], columns=columns)
        df = df[df[columns[0]] != columns[0]]
        df.reset_index(drop=True, inplace=True)

        cleaned_rows = []
        for idx, row in df.iterrows():
            if idx == 0:
                cleaned_rows.append(row)
                continue
            if not row[columns[0]] or str(row[columns[0]]).strip() == "":
                if cleaned_rows and len(columns) > 4:
                    cleaned_rows[-1][columns[4]] = (
                        str(cleaned_rows[-1][columns[4]]) + " " + str(row[columns[4]])
                    )
            else:
                cleaned_rows.append(row)

        df_cleaned = pd.DataFrame(cleaned_rows)
        df_cleaned.reset_index(drop=True, inplace=True)
        return df_cleaned


# ===========================================================================
# Generic Parser (table-based catch-all)
# ===========================================================================
class GenericParser(BaseParser):
    name = "GenericParser"

    def parse(self, pdf_path: str, password: Optional[str] = None) -> pd.DataFrame:
        all_rows = []
        seen_header = None

        with pdfplumber.open(pdf_path, password=password) as pdf:
            for page in pdf.pages:
                table = page.extract_table()
                if not table:
                    continue
                for row in table:
                    if not any(row):
                        continue
                    row_lower = [c.strip().lower() if c else "" for c in row]
                    if seen_header is None:
                        seen_header = row_lower
                        all_rows.append(row)
                    elif row_lower != seen_header:
                        all_rows.append(row)

        if not all_rows or len(all_rows) < 2:
            raise ValueError("GenericParser: No valid table found in PDF.")

        df = pd.DataFrame(all_rows[1:], columns=all_rows[0])
        for col in df.columns:
            df[col] = df[col].apply(clean_numeric)
        return df


# ===========================================================================
# Parser Factory with Fallback Chain
# ===========================================================================

# Singleton instances
_hdfc_v1    = HDFCParser_v1()
_union      = UnionBankParser()
_canara     = CanaraParser()
_generic    = GenericParser()

# Primary + ordered fallback chain per bank key
PARSER_CHAINS: dict[str, list[BaseParser]] = {
    "hdfc":      [_hdfc_v1, _generic],
    "unionbank": [_union,   _generic],
    "canara":    [_canara,  _generic],
    # Unrecognised banks → generic only
    "others":    [_generic],
    "generic":   [_generic],
    # New banks that share the generic table layout
    "icici":     [_generic],
    "sbi":       [_generic],
    "axis":      [_generic],
    "kotak":     [_generic],
    "pnb":       [_generic],
    "bob":       [_generic],
}


def run_parser_chain(bank_key: str, pdf_path: str, password: Optional[str]) -> tuple:
    """
    Tries each parser in the chain for the given bank key.
    Returns (DataFrame, page_count, parser_name_used).
    Raises RuntimeError if all parsers fail.
    """
    chain = PARSER_CHAINS.get(bank_key, [_generic])
    last_error = None

    for parser in chain:
        try:
            logger.info("Attempting %s ...", parser.name)
            df, page_count, name = parser.run(pdf_path, password)
            if not df.empty:
                logger.info("Success with %s — %d rows", name, len(df))
                return df, page_count, name
            logger.warning("%s returned empty DataFrame, trying next.", name)
        except Exception as e:
            last_error = e
            logger.warning("%s failed: %s. Trying next parser.", parser.name, e)

    raise RuntimeError(
        f"All parsers failed for bank '{bank_key}'. "
        f"Last error: {last_error}"
    )


# ===========================================================================
# Auto Bank Detection (enhanced)
# ===========================================================================
BANK_KEYWORDS: dict[str, list[str]] = {
    "hdfc":      ["hdfc bank", "hdfc"],
    "unionbank": ["union bank of india", "union bank"],
    "canara":    ["canara bank"],
    "icici":     ["icici bank"],
    "sbi":       ["state bank of india", "sbi"],
    "axis":      ["axis bank"],
    "kotak":     ["kotak mahindra", "kotak bank"],
    "pnb":       ["punjab national bank", "pnb"],
    "bob":       ["bank of baroda"],
}

def detect_bank(pdf_path: str, password: Optional[str] = None) -> str:
    """
    Auto-detect bank by scanning the first two pages for keywords.
    Returns a bank key string (e.g. 'hdfc') or 'generic' as fallback.
    """
    try:
        with pdfplumber.open(pdf_path, password=password) as pdf:
            pages_to_check = pdf.pages[:2]
            full_text = " ".join(
                (page.extract_text() or "") for page in pages_to_check
            ).lower()

        if not full_text.strip():
            logger.info("Auto-detect: no text found — defaulting to generic.")
            return "generic"

        for bank_key, keywords in BANK_KEYWORDS.items():
            if any(kw in full_text for kw in keywords):
                logger.info("Auto-detected bank: %s", bank_key)
                return bank_key

    except Exception as e:
        logger.warning("Auto-detection error: %s", e)

    return "generic"


# ===========================================================================
# Backwards-compatible thin wrappers (legacy API used by old server.py)
# ===========================================================================
def parse_generic(pdf_file, password=None):
    df, _, _ = _generic.run(pdf_file, password)
    return df

def parse_hdfc(pdf_file, password=None):
    df, _, _ = _hdfc_v1.run(pdf_file, password)
    return df

def parse_unionbank(pdf_file, password=None):
    df, _, _ = _union.run(pdf_file, password)
    return df

def parse_canara(pdf_file, password=None):
    df, _, _ = _canara.run(pdf_file, password)
    return df
