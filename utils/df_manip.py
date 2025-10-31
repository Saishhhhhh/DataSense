from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

def get_dataframe_details(df, n_rows: int = 5) -> str:
    return f"""
Columns: {', '.join(df.columns.tolist())}

Data Types:
{df.dtypes.to_string()}

Shape: {df.shape[0]} rows × {df.shape[1]} columns

Sample Data:
{df.head(n_rows).to_string(index=False)}
""".strip()

def manipulate_dataframe_with_llm(model, df, user_request: str):
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

    safe_locals = {"df": df}
    try:
        exec(code, {"__builtins__": {}}, safe_locals)
        new_df = safe_locals["df"]
        return new_df, code, None
    except Exception as e:
        return df, code, str(e)


