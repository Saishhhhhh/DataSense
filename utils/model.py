from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

def get_chat_model(provider: str, model_name: str):
    if provider == "Google Gemini":
        return ChatGoogleGenerativeAI(model=model_name)
    if provider == "OpenAI":
        return ChatOpenAI(model=model_name)
    if provider == "Anthropic Claude":
        return ChatAnthropic(model=model_name)
    raise ValueError("Unsupported provider")


