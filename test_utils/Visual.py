import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create the LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
# System Prompt

BLOCKED_KEYWORDS = [
    "import os", "import sys", "subprocess", "shutil", "open(",
    "socket", "requests", "eval(", "exec(", "os.system", "pip install",
    "__import__", "del ", "input(", "exit(", "quit(", "globals", "locals"
]

def is_code_safe(code: str) -> tuple[bool, str | None]:
    """Check for blacklisted commands in the generated code."""
    for bad in BLOCKED_KEYWORDS:
        if bad.lower() in code.lower():
            return False, f"❌ Unsafe code detected: `{bad}`"
    return True, None

template = PromptTemplate(
    template="""
You are a Python data visualization assistant.

Your task: Generate Python code that creates charts using matplotlib or seaborn for user query for the given dataframe.

dataframe details: {dataframe_details}

Rules:
- DataFrame is always named: df
- ALWAYS import matplotlib.pyplot as plt
- Import seaborn as sns only if needed
- DO NOT modify or recreate `df`
- NEVER write markdown or explanation
- ONLY return raw executable Python code
- Always call plt.show()
- Add titles and axis labels

If the user asks for something unsafe, politely refuse.

user query: {user_query}
""",
input_variables=['dataframe_details','user_query']
)

parser = StrOutputParser()
query = "2. {'type': 'Horizontal Bar Chart', 'columns': ['country', 'type'], 'description': 'Plot the top N 'country' by the count of titles, potentially faceted or grouped by 'type', to identify major production hubs.'}"

df = pd.read_csv("netflix_titles.csv")

# Create a brief summary of the dataframe
dataframe_details = f"""
Columns: {', '.join(df.columns.tolist())}
dtypes:
{df.dtypes.to_string()}

Sample data:
{df.head(5).to_string()}
"""

# print(dataframe_details)

chain = template | model | parser

plot_code = chain.invoke({
    'dataframe_details': dataframe_details,
    'user_query': query
})

print("Generated Code:\n", plot_code)

# ✅ Safety validation
safe, msg = is_code_safe(plot_code)
if not safe:
    print(msg)
else:
    try:
        # Execute in a restricted namespace
        exec(plot_code, {"pd": pd, "plt": plt, "sns": sns, "df": df}, {})
    except Exception as e:
        print("❌ Error while executing generated code:", e)

