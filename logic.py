import pandas as pd
import streamlit as st

# Usa TU lógica "special" desde logic.py
from logic import (
    invoice_allowed_band,
    target_band_for_new_invoice_from_gr,
    run_analysis_special,
)

st.title("📦 Weight Discrepancy Checker")

st.markdown(
    "<p style='color:#cccccc;'>"
    "Upload the shipment PDFs (1 GR + 1 or more invoices). The system will automatically check for discrepancies."
    "</p>",
    unsafe_allow_html=True
)

# =========================
# Pre-check calculator
# =========================
st.subheader("🧮 Pre-check Calculator")
st.caption(
    "Enter the GR total (kg) and the Invoice total (kg). "
    "This calculates the ±10% allowed band and the target band for a corrected invoice total based on the GR."
)

gr_val = st.number_input("GR (kg):", min_value=0.0, value=0.0, step=0.1)
inv_val = st.number_input("Invoice (kg):", min_value=0.0, value=0.0, step=0.1)

calc = st.button("Calculate")

if calc:
    if gr_val <= 0 or inv_val <= 0:
        st.error("⚠️ Enter values > 0 for GR and Invoice.")
    else:
        low_allowed, high_allowed = invoice_allowed_band(inv_val)
        in_tol = (low_allowed <= gr_val <= high_allowed)

        target_low, target_high = target_band_for_new_invoice_from_gr(gr_val)

        st.markdown(
            "<div style='padding:10px;border:1px solid #ddd;border-radius:8px;'>"
            "<h3 style='margin:0;color:#c00000;'>Weight discrepancy</h3>"
            "<div style='margin-top:6px;'><b>Tolerance:</b> ±10% (allowed band based on the Invoice total)</div>"
            "</div>",
            unsafe_allow_html=True
        )

        df_main = pd.DataFrame([{
            "-10.00% (LOW)": round(low_allowed, 3),
            "Commercial invoice weight -->": round(inv_val, 2),
            "+10.00% (HIGH)": round(high_allowed, 3),
            "enter GXD weight here -->": round(gr_val, 2),
        }])
        st.dataframe(df_main, use_container_width=True)

        st.markdown(
            "<div style='margin-top:10px;padding:10px;border:1px solid #ddd;border-radius:8px;'>"
            "<b>Target band for the NEW Invoice total (based on GR):</b><br>"
            "If there is a discrepancy, the adjustment aims to keep the new invoice total inside this range."
            "</div>",
            unsafe_allow_html=True
        )

        df_target = pd.DataFrame([{
            "Target NEW Invoice LOW (GR/1.10)": round(target_low, 3),
            "Target NEW Invoice HIGH (GR/0.90)": round(target_high, 3),
        }])
        st.dataframe(df_target, use_container_width=True)

        if in_tol:
            st.success("✅ No weight discrepancy (GR is inside the allowed band).")
        else:
            st.warning("⚠️ Weight discrepancy detected. Upload PDFs to run the correction analysis.")

st.divider()

# =========================
# Upload + Run Analysis
# =========================
st.subheader("📤 Shipment PDFs Upload")
st.caption(
    "Upload the shipment PDFs (1 GR + 1 or more invoices). "
    "The system will automatically check for discrepancies."
)

uploaded_files = st.file_uploader(
    "Select PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("🔎 Run Analysis")

if run_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ You must upload at least 2 PDFs: 1 GR and 1 or more invoices.")
    else:
        uploaded = {f.name: f.read() for f in uploaded_files}

        with st.spinner("Analyzing PDFs, please wait…"):
            summary, df_full, df_adjusted, validation_df = run_analysis_special(uploaded, tol=0.10)

        st.success("✅ Analysis completed")

        # =========================
        # Shipment summary
        # =========================
        st.subheader("📊 Shipment Summary")
        st.caption(
            "High-level results: totals, allowed bands, target bands, number of pieces detected, "
            "how many pieces were adjusted, and tolerance status before/after."
        )
        st.dataframe(summary, use_container_width=True)

        # Helpful message based on results
        try:
            in_before = bool(summary.loc[0, "In tolerance BEFORE"])
            in_after = bool(summary.loc[0, "In tolerance AFTER"])
            pieces_changed = int(summary.loc[0, "Pieces changed"])
            if in_before and pieces_changed == 0:
                st.info("No discrepancy was detected, so no piece adjustments were needed.")
            elif (not in_before) and in_after:
                st.success("Discrepancy detected and the adjusted total is now within tolerance.")
            elif (not in_before) and (not in_after):
                st.warning("Discrepancy detected, but the adjusted total is still outside tolerance.")
        except Exception:
            pass

        # =========================
        # Full table (renamed)
        # =========================
        st.subheader("📦 All Pieces Weight Summary (Used for Total Validation)")
        st.caption(
            "Consolidated view of all shipment pieces, including adjusted and non-adjusted cases. "
            "This table is used to verify the total weight and confirm that the shipment no longer has a weight discrepancy."
        )
        st.dataframe(df_full, use_container_width=True)

        if "NEW WEIGHT lbs" in df_full.columns:
            st.write(f"🔹 Sum of NEW WEIGHT lbs: {round(df_full['NEW WEIGHT lbs'].sum(), 2)} lbs")
        if "NEW WEIGHT kgs" in df_full.columns:
            st.write(f"🔹 Sum of NEW WEIGHT kgs: {round(df_full['NEW WEIGHT kgs'].sum(), 2)} kg")

        # =========================
        # Adjusted pieces only
        # =========================
        st.subheader("📦 Adjusted Pieces Only (CAT)")
        st.caption(
            "Only the cases that were adjusted. "
            "Includes the invoice number used for the adjustment and the old vs. new weight values."
        )
        st.dataframe(df_adjusted, use_container_width=True)

        # =========================
        # Validation table
        # =========================
        st.subheader("📊 Validation – Invoice vs GR vs New Weight")
        st.caption(
            "Per-piece validation view: original invoice weight, matched GR weight (if available), "
            "and the resulting new weight used by the adjustment logic."
        )
        st.dataframe(validation_df, use_container_width=True)
