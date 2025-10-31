import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# Initialize LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Load CSV
df = pd.read_csv("netflix_titles.csv")

# -------- Helper to summarize dataframe --------
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

dataframe_details = get_dataframe_details(df)

# -------- Prompt for data manipulation --------
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
    input_variables=["dataframe_details", "user_query"]
)

# -------- User query --------
user_query = "Remove rows with missing director names and keep only title, director, and country columns."

parser = StrOutputParser()

chain = data_manipulation_prompt | model | parser

# -------- Generate pandas code --------
generated_code = chain.invoke({
    "dataframe_details": dataframe_details,
    "user_query": user_query
})

print("🔹 Generated Code:\n")
print(generated_code)

# -------- Execute the generated code safely --------
try:
    # Create a minimal safe namespace
    safe_locals = {"df": df, "pd": pd}

    exec(generated_code, {"__builtins__": {}}, safe_locals)

    # Get updated DataFrame
    df = safe_locals["df"]

    print("\n✅ Data manipulation executed successfully.")
    print("📁 Saved updated dataframe as output.csv")

except Exception as e:
    print("\n❌ Error during code execution:", e)
