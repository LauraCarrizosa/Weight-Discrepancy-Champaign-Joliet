import pandas as pd
import streamlit as st

from logic import (
    invoice_allowed_band,
    target_band_for_new_invoice_from_gr,
    run_analysis_special,
)

st.set_page_config(page_title="Weight Discrepancy Checker", layout="wide")

st.title("📦 Weight Discrepancy Checker")
st.write(
    "Sube los PDFs del shipment (1 GR + 1 o más Invoices). "
    "El sistema hará el chequeo de discrepancias automáticamente."
)

# =========================
# Pre-check Calculator
# =========================
st.subheader("🧮 Pre-check Calculator")
st.caption(
    "Usa esta calculadora para validar si existe discrepancy antes de subir PDFs. "
    "Regla: el GR debe estar dentro de ±10% del total de la Invoice."
)

col1, col2 = st.columns(2)
with col1:
    gr_val = st.number_input("GR (kg)", min_value=0.0, value=0.0, step=0.1)
with col2:
    inv_val = st.number_input("Invoice (kg)", min_value=0.0, value=0.0, step=0.1)

calc_btn = st.button("Calcular")

if calc_btn:
    if gr_val <= 0 or inv_val <= 0:
        st.error("⚠️ Ingresa valores > 0 para GR e Invoice.")
    else:
        low_allowed, high_allowed = invoice_allowed_band(inv_val, tol=0.10)
        in_tol = (low_allowed <= gr_val <= high_allowed)

        target_low, target_high = target_band_for_new_invoice_from_gr(gr_val, tol=0.10)

        df_main = pd.DataFrame([{
            "-10.00% (LOW)": round(low_allowed, 3),
            "Commercial invoice weight -->": round(inv_val, 2),
            "+10.00% (HIGH)": round(high_allowed, 3),
            "enter GXD weight here -->": round(gr_val, 2),
        }])
        st.dataframe(df_main, use_container_width=True)

        df_target = pd.DataFrame([{
            "Target NEW Invoice LOW (GR/1.10)": round(target_low, 3),
            "Target NEW Invoice HIGH (GR/0.90)": round(target_high, 3),
        }])
        st.dataframe(df_target, use_container_width=True)

        if in_tol:
            st.success("✅ NO hay weight discrepancy. No necesitas subir documentos.")
        else:
            st.warning("⚠️ SÍ hay weight discrepancy. Si quieres, sube los PDFs para hacer la corrección.")

st.divider()

# =========================
# Upload + Run Analysis
# =========================
st.subheader("📤 Upload shipment PDFs")
st.caption("Sube mínimo 2 PDFs: 1 GR y 1 o más Invoices.")

uploaded_files = st.file_uploader(
    "Sube los archivos PDF",
    type=["pdf"],
    accept_multiple_files=True
)

run_btn = st.button("🔎 Ejecutar análisis")

if run_btn:
    if not uploaded_files or len(uploaded_files) < 2:
        st.error("⚠️ Debes subir mínimo 2 PDFs: 1 GR + 1 o más Invoices.")
    else:
        uploaded = {f.name: f.read() for f in uploaded_files}

        with st.spinner("Analizando PDFs..."):
            summary, df_full, df_adjusted, validation_df = run_analysis_special(uploaded, tol=0.10)

        st.success("✅ Análisis completado")

        st.subheader("📊 Shipment Summary")
        st.caption(
            "Resumen del shipment: totals, rangos permitidos, target band y estado BEFORE/AFTER."
        )
        st.dataframe(summary, use_container_width=True)

        st.subheader("📦 All Pieces Weight Summary (Used for Total Validation)")
        st.caption(
            "Consolidated view of all shipment pieces, including adjusted and non-adjusted cases. "
            "This table is used to verify the total weight and confirm that the shipment no longer has a weight discrepancy."
        )
        st.dataframe(df_full, use_container_width=True)

        if "NEW WEIGHT lbs" in df_full.columns:
            st.write(f"🔹 Suma NEW WEIGHT lbs: {round(df_full['NEW WEIGHT lbs'].sum(), 2)} lbs")
        if "NEW WEIGHT kgs" in df_full.columns:
            st.write(f"🔹 Suma NEW WEIGHT kgs: {round(df_full['NEW WEIGHT kgs'].sum(), 2)} kg")

        st.subheader("📦 Adjusted Pieces Only (CAT)")
        st.caption(
            "Solo los cases que fueron modificados para llevar el total a tolerancia."
        )
        st.dataframe(df_adjusted, use_container_width=True)

        st.subheader("📊 Validation – Invoice vs GR vs New")
        st.caption(
            "Validación por pieza: peso original invoice vs GR matcheado vs nuevo peso propuesto."
        )
        st.dataframe(validation_df, use_container_width=True)
