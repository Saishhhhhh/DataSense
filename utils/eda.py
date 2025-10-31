import io
import json
import pandas as pd

def run_eda(df: pd.DataFrame) -> dict:
    eda_results = {}
    eda_results["shape"] = df.shape
    eda_results["columns"] = list(df.columns)
    eda_results["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    try:
        info_buf = io.StringIO()
        df.info(buf=info_buf)
        eda_results["memory_usage"] = info_buf.getvalue()
    except Exception:
        eda_results["memory_usage"] = ""

    eda_results["missing_count_per_column"] = df.isnull().sum().to_dict()
    eda_results["missing_percent_per_column"] = (df.isnull().sum() / len(df) * 100).round(3).to_dict()
    eda_results["total_missing_rows"] = int(df.isnull().sum().sum())

    numeric_stats = {}
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        numeric_stats[col] = {
            "mean": float(df[col].mean()) if pd.notnull(df[col].mean()) else None,
            "min": float(df[col].min()) if pd.notnull(df[col].min()) else None,
            "max": float(df[col].max()) if pd.notnull(df[col].max()) else None,
            "std": float(df[col].std()) if pd.notnull(df[col].std()) else None,
            "q1": float(df[col].quantile(0.25)) if pd.notnull(df[col].quantile(0.25)) else None,
            "median": float(df[col].quantile(0.5)) if pd.notnull(df[col].quantile(0.5)) else None,
            "q3": float(df[col].quantile(0.75)) if pd.notnull(df[col].quantile(0.75)) else None,
        }
    eda_results["numeric_stats"] = numeric_stats

    categorical_stats = {}
    for col in df.select_dtypes(include=["object"]).columns:
        value_counts = df[col].value_counts().head()
        categorical_stats[col] = {
            "unique_count": int(df[col].nunique(dropna=True)),
            "top_values": {str(k): int(v) for k, v in value_counts.to_dict().items()},
        }
    eda_results["categorical_stats"] = categorical_stats

    numeric_df = df.select_dtypes(include=["int64", "float64"]).copy()
    try:
        eda_results["correlation_matrix"] = json.loads(numeric_df.corr(numeric_only=True).to_json()) if not numeric_df.empty else {}
    except Exception:
        eda_results["correlation_matrix"] = {}

    eda_results["duplicate_rows"] = int(df.duplicated().sum())
    eda_results["unique_values_per_column"] = {col: int(df[col].nunique(dropna=True)) for col in df.columns}

    outlier_summary = {}
    for col in df.select_dtypes(include=["int64", "float64"]).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        outlier_summary[col] = int(len(outliers))
    eda_results["outliers"] = outlier_summary
    return eda_results


