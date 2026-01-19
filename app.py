import pandas as pd
import streamlit as st

from logic import (
    run_analysis_special,
    invoice_allowed_band,
    target_band_for_new_invoice_from_gr,
)

st.title("📦 Weight Discrepancy Checker Champaign / Joliet")
st.markdown(
    "Sube los PDFs del shipment (1 GR + 1 o más Invoices). "
    "El sistema hará el chequeo de discrepancias automáticamente."
)

# =========================
# Calculadora previa (equivalente a celda 0)
# =========================
gr_val = st.number_input("GR (kg):", min_value=0.0, value=0.0, step=0.1)
inv_val = st.number_input("Invoice (kg):", min_value=0.0, value=0.0, step=0.1)

calc = st.button("Calcular")

if calc:
    if gr_val <= 0 or inv_val <= 0:
        st.error("⚠️ Ingresa valores > 0 para GR e Invoice.")
    else:
        low_allowed, high_allowed = invoice_allowed_band(inv_val)
        in_tol = (low_allowed <= gr_val <= high_allowed)

        target_low, target_high = target_band_for_new_invoice_from_gr(gr_val)

        st.markdown(
            "<div style='padding:10px;border:1px solid #ddd;border-radius:8px;'>"
            "<h3 style='margin:0;color:#c00000;'>Weight discrepancy</h3>"
            "<div style='margin-top:6px;'><b>Tolerancia:</b> ±10% (banda permitida sobre el total de la invoice)</div>"
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
            "<b>Target band para el NUEVO total de Invoice (basado en GR):</b><br>"
            "Si hay discrepancy, el ajuste buscará que el nuevo total de la factura quede dentro de este rango."
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
                "<b style='color:#1b5e20'>✅ NO hay weight discrepancy.</b><br>"
                "No necesitas subir documentos."
                "</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div style='margin-top:10px;padding:10px;border-radius:8px;"
                "background:#fff4e5;border:1px solid #ffb74d;'>"
                "<b style='color:#e65100'>⚠️ SÍ hay weight discrepancy.</b><br>"
                "Si quieres, sube los PDFs para hacer la corrección (se ajusta la factura)."
                "</div>",
                unsafe_allow_html=True
            )

# =========================
# Upload PDFs + Botón ejecutar (equivalente a celdas 2/4)
# =========================
uploaded_files = st.file_uploader(
    "Sube los archivos PDF del shipment",
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

        st.subheader("📊 Resumen del shipment")
        st.dataframe(summary, use_container_width=True)

        st.subheader("📦 Tabla completa (CAT)")
        st.dataframe(df_full, use_container_width=True)
        st.write(f"🔹 Suma NEW WEIGHT kgs: {round(df_full['NEW WEIGHT kgs'].sum(), 2)} kg")

        st.subheader("📦 Solo piezas ajustadas (CAT)")
        st.dataframe(df_adjusted, use_container_width=True)

        st.subheader("📊 Validación – Invoice vs GR vs Nuevo")
        st.dataframe(validation_df, use_container_width=True)


