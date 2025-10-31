import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Create the LLM
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# ---------- STEP 1: Load Data ----------
df = pd.read_csv("sales_data.csv")

# ---------- STEP 2: Helper function ----------
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

# ---------- STEP 3: Define output schema ----------
response_schemas = [
    ResponseSchema(
        name="analytical_questions",
        description="List of 5 insightful natural language queries for analysis."
    ),
    ResponseSchema(
        name="visualization_suggestions",
        description="List of 5 visualizations with chart types and columns to plot."
    )
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

# ---------- STEP 4: Prompt Template ----------
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
4. If possible, align the insights with the domain (e.g., sales, healthcare, movies, finance).

### DataFrame Details:
{dataframe_details}

### Output Format:
{format_instructions}
""",
    input_variables=["dataframe_details", "format_instructions"]
)


# ---------- STEP 5: Run the chain ----------
chain = template | model | parser

result = chain.invoke({
    "dataframe_details": dataframe_details,
    "format_instructions": format_instructions
})


# ---------- STEP 6: Display ----------
print("\n🧠 Analytical Questions:\n")
for i, q in enumerate(result['analytical_questions'], start=1):
    print(f"{i}. {q}")

print("\n📊 Visualization Suggestions:\n")
for i, v in enumerate(result['visualization_suggestions'], start=1):
    print(f"{i}. {v}")
