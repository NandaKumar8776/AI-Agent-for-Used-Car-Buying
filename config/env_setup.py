import os

from dotenv import load_dotenv
load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_PROJECT_NAME"] = os.getenv("LANGCHAIN_PROJECT_NAME")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

os.environ["LANGCHAIN_TRACING_V2"] = "true"


os.environ["SCRAPE_API_KEY"] = os.getenv("SCRAPE_API_KEY")


# Web Crawler URL
crawl_url = "https://www.principleauto.ca/inventory/"

# Prompt locations
os.environ["SUPERVISOR_PROMPT_DIR"] = os.getenv("SUPERVISOR_PROMPT_DIR")
os.environ["RAG_PROMPT_DIR"] = os.getenv("RAG_PROMPT_DIR")
os.environ["LLM_PROMPT_DIR"] = os.getenv("LLM_PROMPT_DIR")
os.environ["WEB_PROMPT_DIR"] = os.getenv("WEB_PROMPT_DIR")