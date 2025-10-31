import os
import io
import json
import traceback
import pandas as pd
import streamlit as st

# Optional plotting libs for visualization tab
import matplotlib.pyplot as plt
import seaborn as sns

# LangChain providers
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

# load_dotenv()
# ----------------------------
# Helpers: Suggestion normalization
# ----------------------------
def normalize_nlq_suggestions(suggestions_obj) -> list:
    raw = suggestions_obj or {}
    items = raw.get("analytical_questions", []) if isinstance(raw, dict) else raw
    out = []
    if isinstance(items, list):
        for it in items:
            if isinstance(it, str):
                s = it.strip()
                if s:
                    out.append(s)
            elif isinstance(it, dict):
                # If dict, try common keys
                val = it.get("question") or it.get("text") or it.get("value")
                if isinstance(val, str) and val.strip():
                    out.append(val.strip())
    elif isinstance(items, str):
        # Attempt to split lines/bullets
        for line in items.splitlines():
            s = line.strip(" -•\t\r\n")
            if len(s) > 1:
                out.append(s)
    # Ensure at most 5, remove empties and dups
    dedup = []
    seen = set()
    for s in out:
        if s and s not in seen:
            seen.add(s)
            dedup.append(s)
    return dedup[:5]


def normalize_viz_suggestions(suggestions_obj) -> list:
    raw = suggestions_obj or {}
    items = raw.get("visualization_suggestions", []) if isinstance(raw, dict) else raw
    normalized = []
    def to_prompt(obj):
        if not obj:
            return None
        if isinstance(obj, str):
            return obj.strip()
        if isinstance(obj, dict):
            t = obj.get("type")
            desc = obj.get("description") or obj.get("desc") or obj.get("text")
            cols = obj.get("columns") or obj.get("cols") or []
            cols_txt = ", ".join([str(c) for c in cols]) if cols else "relevant columns"
            if t and desc:
                return f"Create a {t} — {desc} Use columns {cols_txt}."
            if desc:
                return f"{desc}"
            return None
        return None

    if isinstance(items, list):
        for it in items:
            if isinstance(it, dict):
                prompt = to_prompt(it)
                normalized.append({
                    "type": it.get("type") or "Visualization",
                    "description": it.get("description") or it.get("desc") or it.get("text") or "",
                    "columns": it.get("columns") or it.get("cols") or [],
                    "prompt": prompt or "",
                })
            else:
                s = str(it).strip()
                if s:
                    normalized.append({
                        "type": "Visualization",
                        "description": s,
                        "columns": [],
                        "prompt": s,
                    })
    elif isinstance(items, str):
        s = items.strip()
        if s:
            normalized.append({
                "type": "Visualization",
                "description": s,
                "columns": [],
                "prompt": s,
            })

    # Trim to 5 and clean up
    out = []
    seen_prompts = set()
    for obj in normalized:
        p = obj.get("prompt", "").strip()
        if p and p not in seen_prompts:
            seen_prompts.add(p)
            out.append(obj)
    return out[:5]


# ----------------------------
# Helpers: Code safety for viz (aligns with Visual.py intent)
# ----------------------------
BLOCKED_KEYWORDS = [
    "import os", "import sys", "subprocess", "shutil", "open(",
    "socket", "requests", "eval(", "exec(", "os.system", "pip install",
    "__import__", "del ", "input(", "exit(", "quit(", "globals", "locals"
]

def is_code_safe(code: str) -> tuple[bool, str | None]:
    for bad in BLOCKED_KEYWORDS:
        if bad.lower() in code.lower():
            return False, f"Unsafe code detected: `{bad}`"
    return True, None


# ----------------------------
# Helpers: Model selection
# ----------------------------
def get_chat_model(provider: str, model_name: str):
    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(model=model_name)
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name)
    if provider == "Anthropic Claude":
        return ChatAnthropic(model=model_name)
    raise ValueError("Unsupported provider")


# ----------------------------
# Helpers: Data loading
# ----------------------------
def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    filename = uploaded_file.name.lower()
    if filename.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    if filename.endswith(".xlsx") or filename.endswith(".xls"):
        return pd.read_excel(uploaded_file)
    raise ValueError("Unsupported file format. Please upload CSV or Excel.")


# ----------------------------
# EDA logic (mirrors EDA.py without altering it)
# ----------------------------
def run_eda(df: pd.DataFrame) -> dict:
    eda_results = {}
    eda_results["shape"] = df.shape
    eda_results["columns"] = list(df.columns)
    eda_results["dtypes"] = {k: str(v) for k, v in df.dtypes.to_dict().items()}
    try:
        # memory_usage_string is not a public API; fallback gracefully
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

    # IQR outliers
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


# ----------------------------
# Dataframe details string (used by LLM prompts)
# ----------------------------
def get_dataframe_details(df: pd.DataFrame, n_rows: int = 5) -> str:
    details = f"""
Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

Shape: {df.shape[0]} rows × {df.shape[1]} columns

Sample Data:
{df.head(n_rows).to_string(index=False)}
"""
    return details.strip()


# ----------------------------
# Insight suggestion chain (mirrors Insight_suggestor.py)
# ----------------------------
def get_insight_suggestions(model, df: pd.DataFrame):
    dataframe_details = get_dataframe_details(df)
    response_schemas = [
        ResponseSchema(name="analytical_questions", description="List of 5 insightful natural language queries for analysis."),
        ResponseSchema(name="visualization_suggestions", description="List of 5 visualizations with chart types and columns to plot."),
    ]
    parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = parser.get_format_instructions()
    template = PromptTemplate(
        template="""
You are a skilled Python data analyst and EDA expert.

Your job is to carefully study the given dataframe details and suggest useful **analytical questions** and **visualizations** 
that can help a data analyst gain deeper insights into this dataset.

### Instructions:
1. Understand the dataframe details (column names, datatypes, and example values if present).
2. Suggest exactly **5 insightful analytical questions** that can be answered using the data.
   - These should sound like natural language queries (NLQ), not SQL or code.
3. Suggest exactly **5 visualizations** that would reveal key patterns.
   - Mention the type of visualization (e.g., bar chart, boxplot, scatter plot, line chart, heatmap).
   - Specify which columns or relationships to visualize.
4. If possible, align the insights with the domain.

### DataFrame Details:
{dataframe_details}

### Output Format:
{format_instructions}
""",
        input_variables=["dataframe_details", "format_instructions"],
    )
    chain = template | model | parser
    result = chain.invoke({"dataframe_details": dataframe_details, "format_instructions": format_instructions})
    return result


# ----------------------------
# NLQ answering: use agent executor to execute pandas code (like NLQ.ipynb)
# ----------------------------
def answer_nlq_text(model, df: pd.DataFrame, question: str) -> str:
    system_prompt = """
You are a safe data analysis assistant. 
You are allowed to manipulate data using pandas operations like filtering, grouping, sorting, merging, etc.
You must **not** execute or suggest any commands that:
- read, write, or delete files other than explicitly mentioned CSV outputs
- import or use system libraries (os, sys, subprocess, shutil, socket, requests)
- run shell commands, install packages, or use eval/exec
- access the internet or external resources

If the user asks for something unsafe, politely refuse.
When answering, provide specific numbers and results from the data, not approximations.
"""
    
    try:
        # Create the dataframe agent with safety instructions (mirrors NLQ.ipynb)
        agent = create_pandas_dataframe_agent(
            model,
            df,
            verbose=False,  # Set to True if you want to see tool invocations
            allow_dangerous_code=True,  # Required for pandas agent
            agent_type="openai-tools",  # Ensures reasoning with tool use
            prefix=system_prompt,
        )
        result = agent.invoke(question)
        # Agent returns a dict with 'input' and 'output' keys
        if isinstance(result, dict):
            return result.get("output", str(result))
        return str(result)
    except Exception as e:
        return f"Error executing NLQ: {str(e)}"


# ----------------------------
# Visualization generation: produce pyplot/seaborn code and execute safely
# ----------------------------
def generate_and_render_chart(model, df: pd.DataFrame, viz_request: str):
    details = get_dataframe_details(df)
    prompt = PromptTemplate(
        template="""
You are a Python visualization assistant.
Generate ONLY executable matplotlib/seaborn code to create the requested visualization from DataFrame `df`.

Rules:
- Use `import matplotlib.pyplot as plt` and `import seaborn as sns` ONLY if needed inside code.
- Do NOT read/write files. Do NOT show() the plot. Do NOT print.
- Always create a figure and axis: `fig, ax = plt.subplots(figsize=(8,5))` and plot on `ax`.
- Title and label axes when sensible.

### DataFrame Details:
{details}

### Visualization Request:
{viz_request}

Output only raw code, no markdown.
""",
        input_variables=["details", "viz_request"],
    )
    chain = prompt | model | StrOutputParser()
    code = chain.invoke({"details": details, "viz_request": viz_request})

    # Remove any import statements to avoid blocked imports in restricted exec
    sanitized_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        sanitized_lines.append(line)
    code = "\n".join(sanitized_lines)

    # Safety check like Visual.py
    ok, msg = is_code_safe(code)
    if not ok:
        return None, code, msg

    # Execute safely
    safe_builtins = {
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "abs": abs,
        "round": round,
        "int": int,
        "float": float,
        "str": str,
        "list": list,
        "dict": dict,
        "set": set,
        "tuple": tuple,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
    }
    safe_globals = {"__builtins__": safe_builtins}
    # Provide a default fig/ax for code that references ax without creating it
    default_fig, default_ax = plt.subplots(figsize=(8, 5))
    safe_locals = {"df": df, "pd": pd, "plt": plt, "sns": sns, "fig": default_fig, "ax": default_ax}
    # Ensure a fresh figure context per run
    plt.close("all")
    try:
        exec(code, safe_globals, safe_locals)
        # Try to get fig from locals (recommended), else fallback to current figure
        fig = safe_locals.get("fig", plt.gcf())
        return fig, code, None
    except Exception as e:
        return None, code, str(e)


# ----------------------------
# Dataframe manipulation: reuse core prompt/guardrails from dataframe_manipulation.py
# ----------------------------
def manipulate_dataframe_with_llm(model, df: pd.DataFrame, user_request: str):
    dataframe_details = get_dataframe_details(df)
    data_manipulation_prompt = PromptTemplate(
        template="""
You are a **safe Python data manipulation assistant**.

The current DataFrame is named `df`.

Your job:
Generate **only executable pandas code** that performs the user request.

### Rules:
- You may use: filtering, grouping, sorting, adding/removing columns, renaming, merging, etc.
- You MUST NOT import or use modules like os, sys, subprocess, shutil, socket, or requests.
- DO NOT perform file I/O except saving the final DataFrame as `output.csv`.
- DO NOT use eval(), exec(), or shell commands.
- DO NOT print anything or explain steps — only output pure Python code.
- The final code must always:
  1. Modify or create a new DataFrame (still named `df`).
  2. Save it using `df.to_csv('output.csv', index=False)`

### DataFrame Details:
{dataframe_details}

### User Request:
{user_query}

Output only raw code — no markdown, no explanation.
""",
        input_variables=["dataframe_details", "user_query"],
    )
    chain = data_manipulation_prompt | model | StrOutputParser()
    code = chain.invoke({"dataframe_details": dataframe_details, "user_query": user_request})

    safe_locals = {"df": df, "pd": pd}
    try:
        exec(code, {"__builtins__": {}}, safe_locals)
        new_df = safe_locals["df"]
        return new_df, code, None
    except Exception as e:
        return df, code, str(e)


# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(
    page_title="DataSense - AI-Powered Data Analysis",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("📊 DataSense - AI-Powered Data Analysis")

# Sidebar: BYOK + model selection
with st.sidebar:
    st.header("Settings")
    provider = st.selectbox(
        "Model Provider",
        ["Google Gemini", "OpenAI", "Anthropic Claude"],
        index=0,
    )
    if provider == "Google Gemini":
        default_model = "gemini-2.5-flash"
        key_label = "GOOGLE_API_KEY"
    elif provider == "OpenAI":
        default_model = "gpt-4o-mini"
        key_label = "OPENAI_API_KEY"
    else:
        default_model = "claude-3-5-sonnet-latest"
        key_label = "ANTHROPIC_API_KEY"

    api_key = st.text_input(f"{key_label}", type="password", value="", help="Enter your API key here")
    model_name = st.text_input("Model Name", value=default_model)
    
    # Only use input field - no env fallback
    if api_key:
        os.environ[key_label] = api_key
        api_key_to_use = api_key
    else:
        api_key_to_use = None
        st.warning(f"⚠️ Please enter your {key_label} to use AI features.")
    st.divider()
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])

# Store DF in session
if "df" not in st.session_state:
    st.session_state.df = None

if uploaded:
    try:
        st.session_state.df = load_uploaded_file(uploaded)
    except Exception as e:
        st.error(f"Failed to read file: {e}")

df = st.session_state.df

if df is None:
    st.info("Upload a CSV or Excel file to begin.")
    st.stop()

# Instantiate model (only if API key is available)
if not api_key_to_use:
    st.error(f"❌ {key_label} is required to use AI features. Please enter your API key in the sidebar.")
    st.stop()

try:
    chat_model = get_chat_model(provider, model_name)
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()

# Handle prefills before widgets are created
nlq_prefill = st.session_state.pop("_nlq_prefill", None)
if nlq_prefill is not None:
    st.session_state["nlq_question"] = nlq_prefill

viz_prefill = st.session_state.pop("_viz_prefill", None)
if viz_prefill is not None:
    st.session_state["viz_request"] = viz_prefill

tab_eda, tab_nlq, tab_viz, tab_manip = st.tabs(["EDA", "NLQ", "Visualization", "Dataframe Manipulator"])

with tab_eda:
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.markdown("---")
    
    # Overview Metrics
    results = run_eda(df)
    shape = results["shape"]
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📊 Total Rows", f"{shape[0]:,}", help="Total number of rows in the dataset")
    with col2:
        st.metric("📋 Total Columns", shape[1], help="Total number of columns in the dataset")
    with col3:
        st.metric("🔢 Duplicate Rows", f"{results['duplicate_rows']:,}", help="Number of duplicate rows found")
    with col4:
        missing_total = results.get("total_missing_rows", 0)
        missing_pct = (missing_total / (shape[0] * shape[1]) * 100) if shape[0] > 0 else 0
        st.metric("⚠️ Missing Values", f"{missing_total:,}", f"{missing_pct:.2f}%", help="Total missing values across all columns")
    
    st.markdown("---")
    
    # DataFrame Preview
    st.markdown("### 📋 Data Preview")
    st.dataframe(df.head(20), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Column Information
    st.markdown("### 📑 Column Information")
    col_info_df = pd.DataFrame({
        "Column": results["columns"],
        "Data Type": [results["dtypes"].get(col, "Unknown") for col in results["columns"]],
        "Unique Values": [results["unique_values_per_column"].get(col, 0) for col in results["columns"]],
        "Missing Count": [results["missing_count_per_column"].get(col, 0) for col in results["columns"]],
        "Missing %": [f"{results['missing_percent_per_column'].get(col, 0):.2f}%" for col in results["columns"]]
    })
    st.dataframe(col_info_df, use_container_width=True, height=300)
    
    st.markdown("---")
    
    # Missing Values Visualization
    if missing_total > 0:
        st.markdown("### 📉 Missing Values Analysis")
        missing_data = {
            "Column": list(results["missing_count_per_column"].keys()),
            "Missing Count": list(results["missing_count_per_column"].values()),
            "Missing %": [results["missing_percent_per_column"].get(col, 0) for col in results["missing_count_per_column"].keys()]
        }
        missing_df = pd.DataFrame(missing_data)
        missing_df = missing_df[missing_df["Missing Count"] > 0].sort_values("Missing Count", ascending=False)
        
        if not missing_df.empty:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            fig, ax = plt.subplots(figsize=(10, max(6, len(missing_df) * 0.5)))
            sns.barplot(data=missing_df, y="Column", x="Missing %", ax=ax, palette="viridis")
            ax.set_title("Missing Values Percentage by Column", fontsize=14, fontweight="bold")
            ax.set_xlabel("Missing Percentage (%)", fontsize=12)
            ax.set_ylabel("Column", fontsize=12)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
    
    # Numeric Statistics
    if results["numeric_stats"]:
        st.markdown("### 🔢 Numeric Columns Statistics")
        numeric_cols = list(results["numeric_stats"].keys())
        
        for col in numeric_cols:
            stats = results["numeric_stats"][col]
            with st.expander(f"📊 {col} - Statistical Summary", expanded=False):
                stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
                with stat_col1:
                    st.metric("Mean", f"{stats['mean']:,.2f}" if stats['mean'] is not None else "N/A")
                    st.metric("Median", f"{stats['median']:,.2f}" if stats['median'] is not None else "N/A")
                with stat_col2:
                    st.metric("Min", f"{stats['min']:,.2f}" if stats['min'] is not None else "N/A")
                    st.metric("Max", f"{stats['max']:,.2f}" if stats['max'] is not None else "N/A")
                with stat_col3:
                    st.metric("Std Dev", f"{stats['std']:,.2f}" if stats['std'] is not None else "N/A")
                    outliers = results["outliers"].get(col, 0)
                    st.metric("Outliers", f"{outliers:,}")
                with stat_col4:
                    st.metric("Q1", f"{stats['q1']:,.2f}" if stats['q1'] is not None else "N/A")
                    st.metric("Q3", f"{stats['q3']:,.2f}" if stats['q3'] is not None else "N/A")
        
        # Correlation Matrix Visualization
        if results["correlation_matrix"] and len(numeric_cols) > 1:
            st.markdown("### 🔗 Correlation Matrix")
            try:
                corr_df = pd.DataFrame(results["correlation_matrix"])
                # Ensure numeric types - convert object dtype to numeric
                corr_df = corr_df.apply(pd.to_numeric, errors='coerce')
                # Fill NaN values with 0 (for missing correlations)
                corr_df = corr_df.fillna(0)
                
                if not corr_df.empty and corr_df.shape[0] > 0:
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="coolwarm", center=0, 
                               square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax,
                               vmin=-1, vmax=1)
                    ax.set_title("Correlation Matrix", fontsize=14, fontweight="bold", pad=20)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close()
            except Exception as e:
                st.warning(f"Could not display correlation matrix: {str(e)}")
    
    # Categorical Statistics
    if results["categorical_stats"]:
        st.markdown("### 📝 Categorical Columns Statistics")
        cat_cols = list(results["categorical_stats"].keys())
        
        for col in cat_cols:
            stats = results["categorical_stats"][col]
            with st.expander(f"📊 {col} - Top Values", expanded=False):
                st.metric("Unique Count", f"{stats['unique_count']:,}")
                if stats["top_values"]:
                    top_vals_df = pd.DataFrame({
                        "Value": list(stats["top_values"].keys()),
                        "Count": list(stats["top_values"].values())
                    }).sort_values("Count", ascending=False)
                    st.dataframe(top_vals_df, use_container_width=True, height=200)
                    
                    # Visualize top values
                    if len(top_vals_df) > 0:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        top_10 = top_vals_df.head(10)
                        sns.barplot(data=top_10, x="Count", y="Value", ax=ax, hue="Value", palette="mako", legend=False)
                        ax.set_title(f"Top 10 Values in {col}", fontsize=14, fontweight="bold")
                        ax.set_xlabel("Count", fontsize=12)
                        ax.set_ylabel("Value", fontsize=12)
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close()
    
    st.markdown("---")
    
    # Download Section
    st.markdown("### 💾 Export Results")
    eda_json = json.dumps(results, indent=2)
    st.download_button(
        "📥 Download EDA Report (JSON)", 
        data=eda_json, 
        file_name="eda_results.json", 
        mime="application/json",
        use_container_width=True
    )

with tab_nlq:
    st.subheader("Natural Language Query")
    col1, col2 = st.columns([2, 1])
    with col1:
        question = st.text_area("Ask a question about your data", height=100, key="nlq_question")
        if st.button("Run NLQ") and question.strip():
            with st.spinner("Thinking..."):
                answer = answer_nlq_text(chat_model, df, question)
            st.markdown("**Answer:**")
            # Use st.write for better markdown/text rendering
            st.write(answer)
    with col2:
        st.markdown("**Suggestions**")
        if st.button("Generate NLQ Suggestions"):
            with st.spinner("Generating suggestions..."):
                try:
                    st.session_state["_nlq_suggestions"] = get_insight_suggestions(chat_model, df)
                except Exception as e:
                    st.warning(f"Suggestion generation failed: {e}")
        suggestions = st.session_state.get("_nlq_suggestions")
        if suggestions:
            nlq_list = normalize_nlq_suggestions(suggestions)
            for i, q in enumerate(nlq_list, start=1):
                label = q if len(q) <= 80 else (q[:77] + "...")
                if st.button(f"Q{i}: {label}"):
                    st.session_state["_nlq_prefill"] = q
                    st.rerun()
        if "_nlq_last_answer" in st.session_state:
            st.markdown("**Suggested Answer:**")
            st.write(st.session_state["_nlq_last_answer"])

with tab_viz:
    st.subheader("Visualization")
    
    # Input field and button at the top
    viz_req = st.text_area("Describe the chart you want to create", height=100, key="viz_request")
    col_btn1, col_btn2 = st.columns([1, 10])
    with col_btn1:
        run_viz = st.button("Generate Chart", use_container_width=True)
    
    # Chart generation right below input - immediate display
    if run_viz and viz_req and viz_req.strip():
        with st.spinner("Generating chart..."):
            fig, code, err = generate_and_render_chart(chat_model, df, viz_req)
        
        if err:
            st.error(f"Execution error: {err}")
        else:
            st.pyplot(fig, clear_figure=True)
        
        with st.expander("📝 View Generated Code", expanded=False):
            st.code(code, language="python")
    
    st.markdown("---")
    
    # Generate Suggestions button - outside expander
    if st.button("💡 Generate Visualization Suggestions", use_container_width=True):
        with st.spinner("Generating suggestions..."):
            try:
                st.session_state["_viz_suggestions"] = get_insight_suggestions(chat_model, df)
                st.success("Suggestions generated! Expand below to see them.")
            except Exception as e:
                st.error(f"Suggestion generation failed: {e}")
    
    # Suggestions section below - collapsible
    with st.expander("💡 Visualization Suggestions", expanded=True):
        viz_suggestions = st.session_state.get("_viz_suggestions")
        if viz_suggestions:
            cards = normalize_viz_suggestions(viz_suggestions)
            for i, item in enumerate(cards, start=1):
                with st.container():
                    # Extract chart type and clean description
                    chart_type = item.get("type", "Visualization")
                    desc = item.get("description") or ""
                    
                    # Clean description - remove chart type prefix if present
                    desc_clean = str(desc).strip()
                    
                    # Remove chart type from beginning of description (case-insensitive)
                    # Handle patterns like "Stacked Area Chart: Description" or "Stacked Area Chart Description"
                    type_variants = [
                        chart_type,
                        chart_type.replace(" Chart", ""),
                        chart_type.replace(" Charts", ""),
                        chart_type.replace("Bar Chart", "Bar"),
                        chart_type.replace("Line Chart", "Line"),
                    ]
                    
                    for variant in type_variants:
                        # Pattern: "Type: Description"
                        if desc_clean.lower().startswith(variant.lower() + ":"):
                            desc_clean = desc_clean[len(variant) + 1:].strip()
                            break
                        # Pattern: "Type Description" (when type ends with chart word)
                        elif desc_clean.lower().startswith(variant.lower() + " "):
                            desc_clean = desc_clean[len(variant):].strip()
                            break
                    
                    # Display in a cleaner format with individual button for each
                    col_title, col_btn = st.columns([4, 1])
                    with col_title:
                        st.markdown(f"**V{i}: {chart_type}**")
                        if desc_clean:
                            st.markdown(f"<div style='margin: 0.25em 0 0.5em 1em; color: #555; line-height: 1.5; font-size: 0.9em;'>{desc_clean}</div>", unsafe_allow_html=True)
                        cols = item.get("columns") or []
                        if cols:
                            st.caption(f"📊 Columns: {', '.join(map(str, cols))}")
                    with col_btn:
                        if st.button("Use", key=f"use_viz_{i}", use_container_width=True):
                            chosen = item.get("prompt") or item.get("description") or ""
                            st.session_state["_viz_prefill"] = chosen
                            st.rerun()
                    
                    if i < len(cards):
                        st.divider()
        else:
            st.info("Click 'Generate Visualization Suggestions' above to get started.")

with tab_manip:
    st.subheader("Dataframe Manipulator")
    st.markdown("Enter a manipulation request. The assistant will generate pandas code, execute it, and save `output.csv`.")
    req = st.text_area("Manipulation request", placeholder="e.g., Remove rows with missing director and keep only title, director, country")
    if st.button("Apply Manipulation") and req.strip():
        with st.spinner("Generating and applying code..."):
            new_df, code, err = manipulate_dataframe_with_llm(chat_model, df, req)
        st.markdown("**Generated Code:**")
        st.code(code, language="python")
        if err:
            st.error(f"Execution error: {err}")
        else:
            st.success("Manipulation executed. Preview below and download updated CSV.")
            st.session_state.df = new_df
            df = new_df
    # Always show the latest dataframe preview and download option
    st.dataframe(df.head(200))
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download current dataframe as CSV", data=csv_bytes, file_name="output.csv", mime="text/csv")


