import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Enterprise Server Failure Prediction",
    layout="wide"
)

st.title("Enterprise Server Failure Prediction System")
st.markdown("""
This application predicts whether a server is likely to fail in the next **24 hours**
using monitoring signals such as **memory usage, disk I/O, network latency, errors,
crashes, and maintenance history**.
""")

# =========================
# LOAD MODEL + FEATURE COLUMNS
# =========================
@st.cache_resource
def load_model():
    model = joblib.load("rf_model.pkl")
    feature_cols = joblib.load("feature_columns.pkl")
    return model, feature_cols

model, feature_cols = load_model()

# =========================
# HELPER FUNCTION:
# PREPROCESS + FEATURE ENGINEERING
# =========================
def preprocess_input(df):
    df = df.copy()

    # ---------- 1. Convert timestamp ----------
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    # ---------- 2. Sort ----------
    df = df.sort_values(by=["server_id", "timestamp"]).reset_index(drop=True)

    # ---------- 3. Columns used for monitoring ----------
    monitor_cols = [
        "memory_usage_pct",
        "disk_io_mbps",
        "network_latency_ms",
        "error_count",
        "crash_count"
    ]

    # ---------- 4. Missing indicator columns ----------
    for col in monitor_cols:
        missing_col = col + "_missing"
        df[missing_col] = df[col].isnull().astype(int)

    # ---------- 5. Fill missing values server-wise ----------
    df[monitor_cols] = (
        df.groupby("server_id")[monitor_cols]
          .transform(lambda x: x.ffill().bfill())
    )

    # ---------- 6. Median fallback ----------
    for col in monitor_cols:
        df[col] = df[col].fillna(df[col].median())

    # ---------- 7. Time-based features ----------
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # ---------- 8. Rolling features ----------
    df["mem_avg_3"] = (
        df.groupby("server_id")["memory_usage_pct"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    df["disk_avg_3"] = (
        df.groupby("server_id")["disk_io_mbps"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    df["latency_avg_3"] = (
        df.groupby("server_id")["network_latency_ms"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    df["error_sum_3"] = (
        df.groupby("server_id")["error_count"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    )

    df["crash_sum_3"] = (
        df.groupby("server_id")["crash_count"]
          .transform(lambda x: x.rolling(window=3, min_periods=1).sum())
    )

    # ---------- 9. Trend / diff features ----------
    df["mem_diff"] = df.groupby("server_id")["memory_usage_pct"].diff().fillna(0)
    df["disk_diff"] = df.groupby("server_id")["disk_io_mbps"].diff().fillna(0)
    df["latency_diff"] = df.groupby("server_id")["network_latency_ms"].diff().fillna(0)

    # ---------- 10. Ensure required columns exist ----------
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0

    # ---------- 11. Return model-ready data ----------
    X_input = df[feature_cols].copy()

    return df, X_input


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Upload Input File")
uploaded_file = st.sidebar.file_uploader(
    "Upload server monitoring CSV",
    type=["csv"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Prediction Settings")
threshold = st.sidebar.slider(
    "Failure probability threshold",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

# =========================
# MAIN APP
# =========================
if uploaded_file is not None:
    try:
        # ---------- Read uploaded CSV ----------
        input_df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data Preview")
        st.dataframe(input_df.head())

        # ---------- Preprocess ----------
        processed_df, X_input = preprocess_input(input_df)

        # ---------- Predict ----------
        pred_prob = model.predict_proba(X_input)[:, 1]
        pred_label = (pred_prob >= threshold).astype(int)

        # ---------- Add predictions ----------
        processed_df["failure_probability"] = pred_prob
        processed_df["predicted_failure"] = pred_label

        # =========================
        # METRICS
        # =========================
        total_records = len(processed_df)
        predicted_failures = int(processed_df["predicted_failure"].sum())
        avg_risk = processed_df["failure_probability"].mean()

        col1, col2, col3 = st.columns(3)
        col1.metric("Total Records", f"{total_records:,}")
        col2.metric("Predicted Failures", f"{predicted_failures:,}")
        col3.metric("Average Failure Probability", f"{avg_risk:.2%}")

        # =========================
        # HIGH-RISK SERVERS TABLE
        # =========================
        st.subheader("Top High-Risk Server Records")

        risk_df = processed_df.sort_values(
            by="failure_probability", ascending=False
        )

        display_cols = [
            "timestamp",
            "server_id",
            "client_id",
            "data_center_id",
            "memory_usage_pct",
            "disk_io_mbps",
            "network_latency_ms",
            "error_count",
            "crash_count",
            "days_since_maintenance",
            "failure_probability",
            "predicted_failure"
        ]

        available_display_cols = [c for c in display_cols if c in risk_df.columns]

        st.dataframe(risk_df[available_display_cols].head(50))

        # =========================
        # FILTERED FAILURES ONLY
        # =========================
        st.subheader("Predicted Failure Records Only")
        fail_only_df = risk_df[risk_df["predicted_failure"] == 1]

        if len(fail_only_df) > 0:
            st.dataframe(fail_only_df[available_display_cols].head(100))
        else:
            st.info("No predicted failures for the selected threshold.")

        # =========================
        # CLIENT-WISE RISK SUMMARY
        # =========================
        st.subheader("Client-wise Predicted Failures")

        if "client_id" in processed_df.columns:
            client_summary = (
                processed_df.groupby("client_id")["predicted_failure"]
                .sum()
                .reset_index()
                .sort_values(by="predicted_failure", ascending=False)
            )

            st.dataframe(client_summary)

            fig1, ax1 = plt.subplots(figsize=(8, 4))
            ax1.bar(client_summary["client_id"].astype(str), client_summary["predicted_failure"])
            ax1.set_title("Predicted Failures by Client")
            ax1.set_xlabel("Client ID")
            ax1.set_ylabel("Predicted Failure Count")
            st.pyplot(fig1)

        # =========================
        # DATA CENTER-WISE RISK SUMMARY
        # =========================
        st.subheader("Data Center-wise Predicted Failures")

        if "data_center_id" in processed_df.columns:
            dc_summary = (
                processed_df.groupby("data_center_id")["predicted_failure"]
                .sum()
                .reset_index()
                .sort_values(by="predicted_failure", ascending=False)
            )

            st.dataframe(dc_summary.head(20))

            fig2, ax2 = plt.subplots(figsize=(10, 5))
            top_dc = dc_summary.head(20)
            ax2.bar(top_dc["data_center_id"].astype(str), top_dc["predicted_failure"])
            ax2.set_title("Top 20 Data Centers by Predicted Failure Count")
            ax2.set_xlabel("Data Center ID")
            ax2.set_ylabel("Predicted Failure Count")
            plt.xticks(rotation=45)
            st.pyplot(fig2)

        # =========================
        # FAILURE PROBABILITY DISTRIBUTION
        # =========================
        st.subheader("Failure Probability Distribution")

        fig3, ax3 = plt.subplots(figsize=(8, 4))
        ax3.hist(processed_df["failure_probability"], bins=30)
        ax3.set_title("Distribution of Failure Probability")
        ax3.set_xlabel("Failure Probability")
        ax3.set_ylabel("Count")
        st.pyplot(fig3)

        # =========================
        # DOWNLOAD PREDICTIONS
        # =========================
        st.subheader("Download Predictions")

        csv_data = processed_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download prediction results as CSV",
            data=csv_data,
            file_name="server_failure_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error while processing file: {e}")

else:
    st.info("Upload a CSV file from the sidebar to start predictions.")

    st.markdown("""
    ### Expected input columns
    Your uploaded CSV should ideally contain these columns:

    - `timestamp`
    - `server_id`
    - `client_id`
    - `data_center_id`
    - `memory_usage_pct`
    - `disk_io_mbps`
    - `network_latency_ms`
    - `error_count`
    - `crash_count`
    - `days_since_maintenance`
    - `is_business_hours`
    - `is_weekend`

    The app will automatically:
    - handle missing monitoring values
    - create rolling features
    - create trend features
    - generate failure probability and predicted failure label
    """)
