import os
from utils.helpers import read_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from utils.helpers import llm
from langchain_core.runnables import RunnablePassthrough


class LLMOutput(BaseModel):
    output: str= Field(description="The response of the LLM")


llm_prompt= read_prompt(os.environ["LLM_PROMPT_DIR"])

# Validating the output
parser = JsonOutputParser(pydantic_object= LLMOutput)

prompt = ChatPromptTemplate(
    messages= [
        ("system", llm_prompt),
        ("human", "{user_question}")
    ],
    input_variables= ["user_question"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

# Defining the pipeline
llm_chain_pipeline = (
    {"user_query": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)






