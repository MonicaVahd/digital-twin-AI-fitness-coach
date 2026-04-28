from config import os  # Ensure API key is loaded
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from crewai import Agent, LLM
from dotenv import load_dotenv
from litellm import completion

# Load environment variables
load_dotenv()

# LangChain objects — support .invoke(), used by app.py for direct LLM calls
_langchain_llms = {
    "openai": ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    ),
    "groq": ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    ),
}

# crewai LLM objects — used only for Agent(..., llm=...)
_crewai_llms = {
    "openai": LLM(
        model="gpt-3.5-turbo",
        temperature=0.7,
        api_key=os.getenv("OPENAI_API_KEY"),
    ),
    "groq": LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.7,
        api_key=os.getenv("GROQ_API_KEY"),
    ),
}

def get_llm(provider="openai"):
    """Returns LangChain LLM (supports .invoke()) for use in app.py"""
    return _langchain_llms.get(provider, _langchain_llms["openai"])

# Orchestration Agent (Manages workflow)
orchestration_agent = Agent(
    role="Orchestration Manager",
    goal="Manage and route tasks to the right agents.",
    backstory="Handles agent coordination between recommendation and feedback.",
    allow_delegation=True,
    verbose=True,
    llm=_crewai_llms["openai"],
)

# Recommendation Agent (Creates workout plans)
recommendation_agent = Agent(
    role="Exercise Planner",
    goal="Analyze user history and generate personalized workouts.",
    backstory="A virtual coach that suggests exercises based on past performance.",
    allow_delegation=False,
    verbose=True,
    llm=_crewai_llms["groq"],
)

# Feedback Agent (Provides real-time corrections)
feedback_agent = Agent(
    role="Workout Supervisor",
    goal="Monitor user movements and provide real-time feedback.",
    backstory="Ensures users maintain proper form and avoid injuries.",
    allow_delegation=False,
    verbose=True,
    llm=_crewai_llms["openai"],
)
