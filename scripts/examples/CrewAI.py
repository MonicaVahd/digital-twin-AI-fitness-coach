import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Debug: Print API Key to verify it's loaded
if os.getenv("OPENAI_API_KEY") is None:
    raise ValueError(" ERROR: OPENAI_API_KEY is not set. Please check your .env file.")

print(" LangSmith Tracing:", os.getenv("LANGSMITH_TRACING"))
print(" LangSmith Project:", os.getenv("LANGSMITH_PROJECT"))

# Initialize OpenAI LLM
llm = ChatOpenAI(
    model="gpt-4-turbo",
    temperature=0.7,
    openai_api_key=os.getenv("OPENAI_API_KEY")  # Pass the API key explicitly
)

response = llm.invoke("Hello, world!")
print(response)
