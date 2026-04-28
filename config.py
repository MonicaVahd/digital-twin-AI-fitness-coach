import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Set API key for OpenAI
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")