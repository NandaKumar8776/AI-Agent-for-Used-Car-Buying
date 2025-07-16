import os

## Parser with Pydantic input checking and output checking

from pydantic import BaseModel,Field

class OutputCheck(BaseModel):
    output: str= Field(
        description="The output of the RAG Pipeline, with the context in mind"
    )

from langchain_core.output_parsers import JsonOutputParser

parser = JsonOutputParser(pydantic_object= OutputCheck)

## Prompt Setup

from langchain_core.prompts import ChatPromptTemplate
from utils.helpers import read_prompt

rag_prompt = read_prompt(filepath= os.environ["RAG_PROMPT_DIR"])

prompt = ChatPromptTemplate(
    messages= [
        ("system", rag_prompt),
        ("human","{user_query}"),
    ],
    input_variables=["user_query", "context"],
    partial_variables= {"output_structure": parser.get_format_instructions()}
)

### Hybrid Search RAG Pipeline

from langchain_core.runnables import RunnablePassthrough
from utils.helpers import output_formatter, llm
from langchain.retrievers import EnsembleRetriever

# Importing both the BM25 and Vector Store retrievers
from tools.document_loader import vector_store_retriever, BM25_retriever

# Initialize the ensemble/ hybrid retriever- for BM25 and Flat Indexed L2 Dense Vector Retriever
ensemble_retriever = EnsembleRetriever(retrievers=[BM25_retriever, vector_store_retriever],
                                       weights=[0.3, 0.7])


hybrid_search_rag_pipeline = (
    {"context" : ensemble_retriever | output_formatter, "user_query": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)
