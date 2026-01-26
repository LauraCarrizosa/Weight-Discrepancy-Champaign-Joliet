import re
import io
import pandas as pd
import pdfplumber

# =========================
# CONFIG
# =========================
TOL_DEFAULT = 0.10

DEDUP_CASES_ACROSS_INVOICES = True

BIG_MAX_RATIO = 1.25
BIG_MIN_RATIO = 0.80
TINY_GUARD_KG = 1.0
TINY_MAX_RATIO = 2.0
MAX_ABS_CHANGE_KG = 80.0
CUSHION_KG = 0.5
MIN_CHANGE_KG = 0.30

LBS_TO_KG = 0.45359237
KG_TO_LBS = 1.0 / LBS_TO_KG

PACKING_LIST_RE = re.compile(
    r"(P\s*A\s*C\s*K\s*I\s*N\s*G\s+L\s*I\s*S\s*T)|"
    r"(PACKING\s*LIST)|"
    r"(LISTA\s+DE\s+EMPAQUE)|"
    r"(LISTA\s+DE\s+EMBALAGEM)",
    re.IGNORECASE
)

# =========================
# Bands
# =========================
def invoice_allowed_band(inv_total, tol=0.10):
    # Allowed band based on the INVOICE TOTAL
    return inv_total * (1 - tol), inv_total * (1 + tol)

def target_band_for_new_invoice_from_gr(gr_total, tol=0.10):
    # Target band for the NEW invoice total derived from GR
    return gr_total / (1 + tol), gr_total / (1 - tol)

# =========================
# PDF helpers
# =========================
def pdf_bytes_to_lines(pdf_bytes):
    lines = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for p in pdf.pages:
            txt = p.extract_text() or ""
            for ln in txt.splitlines():
                ln = ln.strip()
                if ln:
                    lines.append(ln)
    return lines

def is_invoice(lines):
    return any(PACKING_LIST_RE.search(l) for l in lines)

def is_gr(lines):
    u = " ".join(lines).upper()
    return ("WAREHOUSE RECEIPT" in u) or ("ORDGR" in u) or re.search(r"\b[A-Z]{3}GR\d{6,}\b", u) is not None

# =========================
# Invoice number extraction (this invoice type)
# =========================
def normalize_invoice_no(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9\s\-]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "").replace("-", "")
    return s

def extract_invoice_number(lines):
    """
    Extracts INVOICE NUMBER for invoices like:
      Header:
        SELLER: BUYER: SHIPPER: INVOICE NUMBER: LOAD NUMBER: DOCPL: ...
      Values:
        Y901   Y30E   92   AC51P216019   044886   1283484 ...
    Returns: AC51P216019
    """

    def split_cols(s: str):
        # NOTE: This expects "column-ish" PDFs (2+ spaces). Works for invoice no in your type.
        return [x.strip() for x in re.split(r"\s{2,}", s.strip()) if x.strip()]

    def is_date_token(tok: str) -> bool:
        return re.match(r"^\d{2}[A-Z]{3}\d{2}$", tok) is not None

    for i in range(len(lines) - 1):
        h = lines[i]
        v = lines[i + 1]

        if "INVOICE NUMBER" not in h.upper():
            continue

        header_cols = split_cols(h.upper())
        value_cols = split_cols(v)

        idx = None
        for k, c in enumerate(header_cols):
            if "INVOICE NUMBER" in c:
                idx = k
                break

        if idx is not None and idx < len(value_cols):
            cell = value_cols[idx].upper().strip()
            toks = re.findall(r"\b[A-Z0-9\-]{6,}\b", cell)
            for t in toks:
                if is_date_token(t):
                    continue
                t_norm = normalize_invoice_no(t)
                if re.match(r"^[A-Z0-9]{6,}$", t_norm):
                    return t_norm

        candidates = re.findall(r"\b[A-Z]{1,5}[A-Z0-9]{5,}\b", v.upper())
        for c in candidates:
            if is_date_token(c):
                continue
            return normalize_invoice_no(c)

    return None

# =========================
# CONTAINER NO extraction (NEW)
# Robust for PDFs where header is "glued" with single spaces.
# Returns e.g. AF260116285
# =========================
def normalize_container_no(s: str) -> str:
    s = (s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9\s\-_/]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    s = s.replace(" ", "").replace("-", "")
    return s

def extract_container_number_special(lines):
    """
    Extract CONTAINER NO for the 'SIN WD' invoice type.

    Why this approach:
    - Some PDFs don't preserve column spacing (2+ spaces). Header comes in a single "sentence".
    - In those cases, "split by columns" fails, so we anchor on "CONTAINER NO" and
      read the value from the next line.
    """

    # 1) Anchor on a line that contains "CONTAINER NO"
    for i in range(len(lines)):
        h = lines[i].upper()
        if "CONTAINER NO" not in h:
            continue

        # 1a) If container appears on same line
        m_same = re.search(r"\bCONTAINER\s+NO\s*[:\-]?\s*([A-Z]{2}\d{6,})\b", h)
        if m_same:
            return normalize_container_no(m_same.group(1))

        # 1b) Normal case: value appears on next line
        if i + 1 < len(lines):
            v = lines[i + 1].upper()

            # Prefer tokens like AF260116285 (2 letters + many digits)
            cands = re.findall(r"\b[A-Z]{2}\d{6,}\b", v)
            if cands:
                # If multiple appear, the right one is often the last.
                return normalize_container_no(cands[-1])

            # Fallback: long tokens, then filter
            toks = re.findall(r"\b[A-Z0-9][A-Z0-9\-_\/]{7,}\b", v)
            for t in reversed(toks):
                if re.match(r"^[A-Z]{2}\d{6,}$", t):
                    return normalize_container_no(t)

    # 2) Fallback global scan near the "CONTAINER NO" occurrence
    text = "\n".join(lines).upper()
    m = re.search(r"\bCONTAINER\s+NO\b", text)
    if m:
        window = text[m.start(): m.start() + 300]
        cands = re.findall(r"\b[A-Z]{2}\d{6,}\b", window)
        if cands:
            return normalize_container_no(cands[-1])

    return None

# =========================
# Extract PACKING LIST block
# =========================
def extract_packing_list_block(lines, invoice_name):
    start = None
    for i, l in enumerate(lines):
        if PACKING_LIST_RE.search(l):
            start = i
            break
    if start is None:
        raise ValueError(f"Invoice '{invoice_name}': PACKING LIST not found.")

    block = []
    for l in lines[start:]:
        u = l.upper().strip()
        if "INVOICE TOTALS" in u:
            break
        if u.startswith("TOTALS:") or u.startswith("TOTALS"):
            break
        block.append(l)

    if not block:
        raise ValueError(f"Invoice '{invoice_name}': PACKING LIST block is empty or malformed.")
    return block

# =========================
# Invoice parser (this invoice type)
# =========================
def parse_invoice_packing_list(lines, invoice_name, invoice_no):
    block = extract_packing_list_block(lines, invoice_name)

    def is_header_line(s):
        u = s.upper()
        return (
            ("GROSS" in u and "NET" in u)
            or ("LBS/" in u and "KILOS" in u)
            or ("PACKAGE" in u and "DESCRIPTION" in u)
            or u.startswith("-" * 5)
        )

    def is_totals_line(s):
        u = s.upper().strip()
        return u.startswith("TOTALS") or u.startswith("TOTALS:")

    piece_line_re = re.compile(r"^\s*(\d{3,})\s+([A-Z0-9]{2,})\s+(.+)$")

    rows = []
    i = 0
    while i < len(block):
        line1 = block[i].strip()
        u1 = line1.upper()

        if is_totals_line(line1):
            break
        if is_header_line(line1):
            i += 1
            continue

        m1 = piece_line_re.match(line1)
        if not m1:
            i += 1
            continue

        pkg_id = m1.group(1)

        if "TOTALS" in u1:
            i += 1
            continue

        nums1 = re.findall(r"\d+(?:\.\d+)?", line1)
        if len(nums1) < 3:
            i += 1
            continue

        gross_lbs = float(nums1[1])

        gross_kg = None
        if i + 1 < len(block):
            line2 = block[i + 1].strip()
            if not is_header_line(line2) and not is_totals_line(line2):
                nums2 = re.findall(r"\d+(?:\.\d+)?", line2)
                if nums2:
                    gross_kg = float(nums2[0])

        if gross_kg is None:
            gross_kg = gross_lbs * LBS_TO_KG

        rows.append({
            "INVOICE_FILE": invoice_name,
            "INVOICE_NO": (invoice_no or invoice_name),
            "CASE_NO": str(pkg_id),
            "CAT_WEIGHT_LBS": gross_lbs,
            "CAT_WEIGHT_KG": gross_kg
        })

        if i + 1 < len(block):
            line2 = block[i + 1].strip()
            nums2 = re.findall(r"\d+(?:\.\d+)?", line2)
            if nums2 and not is_header_line(line2) and not is_totals_line(line2):
                i += 2
                continue

        i += 1

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError(f"Invoice '{invoice_name}': No pieces could be extracted from PACKING LIST.")

    df["CASE_OCC"] = df.groupby(["INVOICE_FILE", "CASE_NO"]).cumcount() + 1
    df["PIECE_ID"] = df["INVOICE_FILE"].astype(str) + "|" + df["CASE_NO"].astype(str) + "|" + df["CASE_OCC"].astype(str)
    return df.reset_index(drop=True)

# =========================
# Dedupe across invoices
# =========================
def collapse_duplicates_across_invoices(inv_df):
    inv_df = inv_df.copy().reset_index(drop=True)

    idx = inv_df.groupby("CASE_NO")["CAT_WEIGHT_KG"].idxmax()
    base = inv_df.loc[idx].copy()

    files = inv_df.groupby("CASE_NO")["INVOICE_FILE"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()
    invnos = inv_df.groupby("CASE_NO")["INVOICE_NO"].apply(lambda s: ", ".join(sorted(set(s)))).reset_index()

    base = base.merge(files, on="CASE_NO", how="left")
    base = base.merge(invnos, on="CASE_NO", how="left", suffixes=("", "_ALL"))

    base["CASE_OCC"] = 1
    base["PIECE_ID"] = base["CASE_NO"].astype(str)
    return base.sort_values("CASE_NO").reset_index(drop=True)

# =========================
# GR parser (KGM) - avoids double counting
# =========================
def parse_gr(lines):
    text = "\n".join(lines).upper()

    header_total = None
    m = re.search(r"/\s*(\d+(?:\.\d+)?)\s*KGM\b", text)
    if m:
        header_total = float(m.group(1))

    piece_weights = []
    piece_line_re = re.compile(r"^\s*(\d{10,})\s+.*?(\d+(?:\.\d+)?)\s*KGM\b", re.IGNORECASE)

    for ln in lines:
        mm = piece_line_re.match(ln.upper())
        if mm:
            piece_weights.append(float(mm.group(2)))

    if piece_weights:
        return float(sum(piece_weights)), piece_weights

    if header_total is not None:
        return float(header_total), [float(header_total)]

    raise ValueError("GR: Could not extract KGM weights (neither per-piece nor header total).")

# =========================
# Similarity matching (sorted-to-sorted)
# =========================
def match_gr_to_invoice_by_similarity(inv_df, gr_pieces):
    inv_sorted = inv_df.sort_values("CAT_WEIGHT_KG").reset_index(drop=True)
    gr_sorted = sorted(gr_pieces)
    n = min(len(inv_sorted), len(gr_sorted))
    mapping = {}
    for i in range(n):
        mapping[inv_sorted.loc[i, "PIECE_ID"]] = float(gr_sorted[i])
    return mapping, len(inv_sorted), len(gr_sorted)

# =========================
# GR-guided adjustment (target band derived from GR)
# NEW KG never exceeds matched GR piece
# =========================
def gr_guided_adjust_auto(inv_df, gr_map, target_low_total, target_high_total):
    inv = inv_df.copy().reset_index(drop=True)
    n = len(inv)
    cur_total = float(inv["CAT_WEIGHT_KG"].sum())

    if target_low_total <= cur_total <= target_high_total:
        return {}

    if cur_total < target_low_total:
        need_up = True
        target = min(target_high_total, target_low_total + CUSHION_KG)
        gap = target - cur_total
    else:
        need_up = False
        target = max(target_low_total, target_high_total - CUSHION_KG)
        gap = cur_total - target

    def allowed_delta(cur, gr):
        if cur <= 0:
            return 0.0

        if gr is not None and isinstance(gr, (int, float)) and gr > 0:
            if need_up:
                return max(0.0, gr - cur)
            else:
                return max(0.0, cur - gr)

        if cur < TINY_GUARD_KG:
            if need_up:
                return cur * (TINY_MAX_RATIO - 1.0)
            else:
                return cur * (1.0 - (1.0 / TINY_MAX_RATIO))
        else:
            if need_up:
                return cur * (BIG_MAX_RATIO - 1.0)
            else:
                return cur * (1.0 - BIG_MIN_RATIO)

    cands = []
    for _, r in inv.iterrows():
        pid = r["PIECE_ID"]
        cur = float(r["CAT_WEIGHT_KG"])
        gr = gr_map.get(pid, None)

        cap = min(MAX_ABS_CHANGE_KG, allowed_delta(cur, gr))
        if cap < MIN_CHANGE_KG:
            continue

        score = (1000 * min(cap, gap)) + (50 * cap) + cur
        cands.append((score, pid, cur, gr, cap))

    cands.sort(key=lambda x: x[0], reverse=True)
    if not cands:
        return {}

    max_by_n = max(3, min(20, int(round(0.50 * n))))

    updates = {}
    remaining = gap

    for _, pid, cur, gr, cap in cands:
        if remaining <= 1e-9:
            break
        if len(updates) >= max_by_n:
            break

        delta = min(remaining, cap)
        if delta < MIN_CHANGE_KG:
            continue

        new_kg = (cur + delta) if need_up else max(0.01, cur - delta)

        if gr is not None and isinstance(gr, (int, float)) and gr > 0:
            new_kg = min(new_kg, gr)

        updates[pid] = round(new_kg, 2)
        remaining -= delta

    return updates

# =========================
# Output tables
# =========================
def build_cat_tables(inv_df, updates):
    rows_full = []
    rows_adj = []

    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        old_lbs = float(r["CAT_WEIGHT_LBS"])
        old_kg = float(r["CAT_WEIGHT_KG"])

        if pid in updates:
            new_kg = float(updates[pid])
            new_lbs = new_kg * KG_TO_LBS
            changed = True
        else:
            new_kg = old_kg
            new_lbs = old_lbs
            changed = False

        rows_full.append({
            "CASE/BOX": r["CASE_NO"],
            "NEW WEIGHT lbs": round(new_lbs, 2),
            "NEW WEIGHT kgs": round(new_kg, 2),
        })

        if changed:
            rows_adj.append({
                "CASE/BOX": r["CASE_NO"],
                "CAT WEIGHT lbs": round(old_lbs, 2),
                "NEW WEIGHT lbs": round(new_lbs, 2),
                "NEW WEIGHT kgs": round(new_kg, 2),
                "NEW LENGTH": "N/A",
                "NEW WIDTH": "N/A",
                "NEW HEIGHT": "N/A",
                "INVOICE #": r.get("INVOICE_NO", r.get("INVOICE_FILE", "")),
            })

    return pd.DataFrame(rows_full), pd.DataFrame(rows_adj)

def build_validation(inv_df, gr_map, updates):
    rows = []
    for _, r in inv_df.iterrows():
        pid = r["PIECE_ID"]
        inv_kg = float(r["CAT_WEIGHT_KG"])
        gr_kg = gr_map.get(pid, None)
        new_kg = float(updates.get(pid, inv_kg))
        rows.append({
            "CASE/BOX": r["CASE_NO"],
            "INVOICE ORIGINAL KG": round(inv_kg, 2),
            "GR KG (matched)": (round(gr_kg, 2) if gr_kg is not None else None),
            "NEW KG": round(new_kg, 2),
        })
    return pd.DataFrame(rows)

# =========================
# RUNNER
# =========================
def run_analysis_special(uploaded, tol=TOL_DEFAULT):
    if len(uploaded) < 2:
        raise ValueError("Upload at least 2 PDFs: 1 GR + 1 or more Invoices.")

    invoices = []
    gr_lines = None
    gr_file = None

    for fname, fbytes in uploaded.items():
        lines = pdf_bytes_to_lines(fbytes)
        if is_gr(lines) and gr_lines is None:
            gr_lines = lines
            gr_file = fname
        else:
            inv_no = extract_invoice_number(lines)
            cntr_no = extract_container_number_special(lines)  # <-- NEW
            invoices.append((fname, lines, inv_no, cntr_no))

    if gr_lines is None:
        raise ValueError("GR not found.")
    if not invoices:
        raise ValueError("No invoice with PACKING LIST was found.")

    gr_total, gr_pieces = parse_gr(gr_lines)

    inv_dfs = [parse_invoice_packing_list(lines, name, inv_no) for name, lines, inv_no, _cntr in invoices]
    inv_df = pd.concat(inv_dfs, ignore_index=True)

    if DEDUP_CASES_ACROSS_INVOICES:
        inv_df = collapse_duplicates_across_invoices(inv_df)

    inv_total = float(inv_df["CAT_WEIGHT_KG"].sum())

    allowed_low_before, allowed_high_before = invoice_allowed_band(inv_total, tol)
    in_before = (allowed_low_before <= gr_total <= allowed_high_before)

    target_low_total, target_high_total = target_band_for_new_invoice_from_gr(gr_total, tol)

    gr_map, _, gr_n = match_gr_to_invoice_by_similarity(inv_df, gr_pieces)

    updates = {}
    if not in_before:
        updates = gr_guided_adjust_auto(inv_df, gr_map, target_low_total, target_high_total)

    df_full, df_adj = build_cat_tables(inv_df, updates)
    new_total = float(df_full["NEW WEIGHT kgs"].sum())

    allowed_low_after, allowed_high_after = invoice_allowed_band(new_total, tol)
    in_after = (allowed_low_after <= gr_total <= allowed_high_after)

    validation_df = build_validation(inv_df, gr_map, updates)

    detected_inv_nos = [x[2] for x in invoices if x[2]]

    # --- NEW: container numbers detected per invoice number (if 2 invoices, show both) ---
    inv_to_cntr = {}
    for _fname, _lines, inv_no, cntr in invoices:
        inv_key = inv_no or "UNKNOWN_INVOICE"
        if cntr:
            inv_to_cntr.setdefault(inv_key, set()).add(cntr)

    if inv_to_cntr:
        containers_detected = " | ".join(
            f"{inv}: {', '.join(sorted(cntrs))}"
            for inv, cntrs in sorted(inv_to_cntr.items())
        )
    else:
        containers_detected = "N/A"

    summary = pd.DataFrame([{
        "GR file": gr_file,
        "Invoices files": ", ".join([x[0] for x in invoices]),
        "Invoice numbers detected": ", ".join(detected_inv_nos) if detected_inv_nos else "NOT_DETECTED",
        "Container numbers detected": containers_detected,  # <-- NEW column

        "Invoice total (kg)": round(inv_total, 2),
        "GR total (kg)": round(gr_total, 2),

        "Allowed low (kg)": round(allowed_low_before, 2),
        "Allowed high (kg)": round(allowed_high_before, 2),

        "Target NEW Invoice low (kg)": round(target_low_total, 2),
        "Target NEW Invoice high (kg)": round(target_high_total, 2),

        "Pieces detected": int(len(inv_df)),
        "GR pieces extracted": int(gr_n),

        "Pieces changed": int(len(df_adj)),
        "New total (kg)": round(new_total, 2),

        "Allowed low after (kg)": round(allowed_low_after, 2),
        "Allowed high after (kg)": round(allowed_high_after, 2),

        "In tolerance BEFORE": bool(in_before),
        "In tolerance AFTER": bool(in_after),
    }])

    return summary, df_full, df_adj, validation_df
