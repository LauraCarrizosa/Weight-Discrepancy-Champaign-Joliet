import pandas as pd
import streamlit as st

from logic import invoice_allowed_band, target_band_for_new_invoice_from_gr, run_analysis

st.title("📦 Weight Discrepancy Checker")
st.markdown(
    "<p style='color:#cccccc;'>"
    "Pre-check calculator used to determine whether a weight discrepancy exists "
    "based on the ±10% tolerance rule, before uploading any documents"
    "</p>",
    unsafe_allow_html=True
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
            "<div style='margin-top:6px;'><b>Tolerance:</b> ±10% (allowed band based on the invoice total)</div>"
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
            "<b>Target band for the NEW invoice total (based on GR):</b><br>"
            "If there is a discrepancy, the adjustment will aim to keep the new invoice total within this range."
            "</div>",
            unsafe_allow_html=True
        )

        df_target = pd.DataFrame([{
            "Target NEW Invoice LOW (GR/1.10)": round(target_low, 3),
            "Target NEW Invoice HIGH (GR/0.90)": round(target_high, 3),
        }])
        st.dataframe(df_target, use_container_width=True)

        if in_tol:
            st.markdown(
                "<div style='margin-top:10px;padding:10px;border-radius:8px;"
                "background:#e7f7e7;border:1px solid #6bbf6b;'>"
                "<b style='color:#1b5e20'>✅ No weight discrepancy.</b><br>"
                "You do not need to upload documents."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='margin-top:10px;padding:10px;border-radius:8px;"
                "background:#fff4e5;border:1px solid #ffb74d;'>"
                "<b style='color:#e65100'>⚠️ Weight discrepancy detected.</b><br>"
                "If you want, upload the PDFs to run the correction (the invoice will be adjusted)."
                "</div>",
                unsafe_allow_html=True
            )

uploaded_files = st.file_uploader(
    "Upload the shipment PDF files (1 GR + 1 or more invoices).",
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
            summary, df_full, df_adjusted, validation_df = run_analysis(uploaded, tol=0.10)

        st.success("✅ Analysis completed")

        st.subheader("📊 Shipment summary")
        st.caption(
            "High-level overview of the shipment: totals, tolerance ranges, and compliance status "
            "before and after the adjustment."
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("📦 All Pieces Weight Summary (Used for Total Validation)")
        st.caption(
        "Consolidated view of all shipment pieces, including adjusted and non-adjusted cases. "
        "This table is used to verify the total weight and confirm that the shipment no longer "
        "has a weight discrepancy."
        )
        st.dataframe(df_full, use_container_width=True)

        st.write(f"🔹 Sum of NEW WEIGHT lbs: {round(df_full['NEW WEIGHT lbs'].sum(), 2)} lbs")
        st.write(f"🔹 Sum of NEW WEIGHT kgs: {round(df_full['NEW WEIGHT kgs'].sum(), 2)} kg")

        st.subheader("📦 Adjusted pieces only (CAT)")
        st.caption(
            "Subset of cases that were adjusted to bring the invoice total within the acceptable tolerance range."
        )
        st.dataframe(df_adjusted, use_container_width=True)

        if validation_df is not None:
            st.subheader("📊 Validation – Invoice vs GR vs New Weight")
            st.caption(
                "Case-level traceability: original invoice weight vs matched GR weight vs the new calculated weight."
            )
            st.dataframe(validation_df, use_container_width=True)



