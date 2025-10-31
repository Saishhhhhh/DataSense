from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

SYSTEM_PROMPT = """
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

def answer_nlq_text(model, df, question: str) -> str:
    agent = create_pandas_dataframe_agent(
        model,
        df,
        verbose=False,
        allow_dangerous_code=True,
        agent_type="openai-tools",
        prefix=SYSTEM_PROMPT,
    )
    result = agent.invoke(question)
    if isinstance(result, dict):
        return result.get("output", str(result))
    return str(result)


