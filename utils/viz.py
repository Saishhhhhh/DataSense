import matplotlib.pyplot as plt
import seaborn as sns
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

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

def get_dataframe_details(df, n_rows: int = 5) -> str:
    return f"""
Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

Shape: {df.shape[0]} rows × {df.shape[1]} columns

Sample Data:
{df.head(n_rows).to_string(index=False)}
""".strip()

def generate_and_render_chart(model, df, viz_request: str):
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

    # Strip imports
    sanitized_lines = []
    for line in code.splitlines():
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            continue
        sanitized_lines.append(line)
    code = "\n".join(sanitized_lines)

    ok, msg = is_code_safe(code)
    if not ok:
        return None, code, msg

    # Allow all builtins except dangerous ones - blacklist approach
    import builtins as original_builtins
    # Create a dict with all safe builtins
    dangerous = {'exec', 'eval', '__import__', 'open', 'file', 'input', 'raw_input', 'exit', 'quit'}
    safe_builtins = {k: v for k, v in original_builtins.__dict__.items() 
                     if k not in dangerous and not k.startswith('__') or k in ['__name__', '__doc__']}
    
    safe_globals = {"__builtins__": safe_builtins}
    import pandas as pd
    fig_default, ax_default = plt.subplots(figsize=(8, 5))
    safe_locals = {"df": df, "pd": pd, "plt": plt, "sns": sns, "fig": fig_default, "ax": ax_default}
    plt.close("all")
    try:
        exec(code, safe_globals, safe_locals)
        fig = safe_locals.get("fig", plt.gcf())
        return fig, code, None
    except Exception as e:
        return None, code, str(e)


