import pandas as pd
import json

df =  pd.read_csv("netflix_titles.csv")

eda_results = {}

eda_results["shape"] = df.shape
eda_results["columns"] = list(df.columns)
eda_results["dtypes"] = df.dtypes.to_dict()
eda_results["memory_usage"] = pd.io.formats.info.DataFrameInfo(df).memory_usage_string.strip()

eda_results["missing_count_per_column"] = df.isnull().sum().to_dict()
eda_results["missing_percent_per_column"] = (df.isnull().sum() / len(df) * 100).to_dict()
eda_results["total_missing_rows"] = df.isnull().sum().sum()

numeric_stats = {}

for col in df.select_dtypes(include=['int64','float64']).columns:
    numeric_stats[col] = {
        "mean": df[col].mean(),
        "min": df[col].min(),
        "max": df[col].max(),
        "std": df[col].std(),
        "q1": df[col].quantile(0.25),
        "median": df[col].quantile(0.5),
        "q3": df[col].quantile(0.75)
    }

eda_results["numeric_stats"] = numeric_stats

categorical_stats = {}

for col in df.select_dtypes(include=['object']).columns:
    categorical_stats[col] = {
        "unique_count": len(df[col].unique()),
        "top_values": df[col].value_counts().head().to_dict()
    }

eda_results["categorical_stats"] = categorical_stats

numeric_df = df.select_dtypes(include=['int64','float64'])
eda_results["correlation_matrix"] = numeric_df.corr().to_dict()

eda_results["duplicate_rows"] = df.duplicated().sum()

eda_results["unique_values_per_column"] = {col: len(df[col].unique()) for col in df.columns}

outlier_summary = {}

for col in df.select_dtypes(include=['int64','float64']):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
    
    outlier_summary[col] = len(outliers)

eda_results["outliers"] = outlier_summary

# eda_results_json = json.dumps(eda_results, indent = 4)
print(eda_results)