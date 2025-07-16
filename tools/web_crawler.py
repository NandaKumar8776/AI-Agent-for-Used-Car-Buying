### Using Scraping Ant Web Scraper
import os
from langchain_community.document_loaders import ScrapingAntLoader
from config.env_setup import crawl_url



print("Webcrawler started- Scraping the webite")

data = ScrapingAntLoader(
    [crawl_url],  # List of URLs to scrape
    api_key=os.environ["SCRAPE_API_KEY"],  # Get your API key from https://scrapingant.com/
    continue_on_failure=True,  # Ignore unprocessable web pages and log their exceptions
).load()

print("Webcrawler started- Scraping completed")


# Setting up the web_crawler 


from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,#hyperparameter
    chunk_overlap=50 #hyperparemeter
)

print("\nScraped data chunking started")

docs = text_splitter.split_documents(data)

print("\nChunking is done")

#### Load documents into the vector store


from memory.vector_store import web_crawler_milvus_vector_store

print("\nUploading the chunked docs to the vector store")
web_crawler_milvus_vector_store.add_documents(
    documents=docs
)

vector_store_retriever = web_crawler_milvus_vector_store.as_retriever()

print("\nVector Store is ready")


#### Load documents into the Keyword Search BM25


from memory.BM25_keyword_search import BM25_retriever

print("\nUploading the web scraped chunked docs to the BM25 Keyword Search")

BM25_web_crawler_retriever= BM25_retriever(docs=docs)

print("\nBM25 Keyword Search is ready")


#### Hybrid Search Web Scraped pipeline

### Hybrid Search RAG Pipeline

from langchain_core.runnables import RunnablePassthrough
from utils.helpers import output_formatter, llm
from langchain.retrievers import EnsembleRetriever


# Initialize the ensemble/ hybrid retriever- for BM25 and Flat Indexed L2 Dense Vector Retriever
ensemble_retriever = EnsembleRetriever(retrievers=[BM25_web_crawler_retriever, vector_store_retriever],
                                       weights=[0.3, 0.7])


import os

## Parser with Pydantic input checking and output checking

from pydantic import BaseModel,Field

class OutputCheck(BaseModel):
    output: str= Field(
        description="The output of the Web Scraper Pipeline, with the context in mind"
    )

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object= OutputCheck)

## Prompt Setup

from langchain_core.prompts import ChatPromptTemplate
from utils.helpers import read_prompt

web_prompt = read_prompt(filepath= os.environ["WEB_PROMPT_DIR"])

prompt = ChatPromptTemplate(
    messages= [
        ("system", web_prompt),
        ("human","{user_query}"),
    ],
    input_variables=["user_query", "context"],
    partial_variables= {"output_structure": parser.get_format_instructions()}
)

hybrid_web_crawl_rag_pipeline = (
    {"context" : ensemble_retriever | output_formatter, "user_query": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)



