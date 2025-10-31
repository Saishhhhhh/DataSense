from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
import pandas as pd

load_dotenv()

# Load your data
df = pd.read_csv("netflix_titles.csv")

# Define a system-style instruction to the model
system_prompt = """
You are a safe data analysis assistant. 
You are allowed to manipulate data using pandas operations like filtering, grouping, sorting, merging, etc.
You must **not** execute or suggest any commands that:
- read, write, or delete files other than explicitly mentioned CSV outputs
- import or use system libraries (os, sys, subprocess, shutil, socket, requests)
- run shell commands, install packages, or use eval/exec
- access the internet or external resources

If the user asks for something unsafe, politely refuse.
"""

# Use the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

# Create the dataframe agent with the safety instruction embedded
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True,
    agent_type="openai-tools",  # ensures reasoning with tool use
    prefix=system_prompt,       # inject safety instructions here
)

# Ask a query
query = "What is the overall distribution of content types (Movies vs. TV Shows) on the platform, and how has this proportion changed year-over-year?"
agent.invoke(query)
