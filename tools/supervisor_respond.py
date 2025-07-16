import os
from utils.helpers import read_prompt
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from utils.helpers import llm
from langchain_core.runnables import RunnablePassthrough


class SupervisorOutput(BaseModel):
    category: Literal["Used Car", "Car Dealership Inventory Website", "Not Related"]= Field(description="The user's question is categorized into these pre-defined categories. Here, Used Car is a Used Car Buying Guide. Car Dealership Inventory Website is the scraped results of a auto dealership's inventory website, scraped. Then, Not Related is anything that is unrelated to those two categories")
    reason: str= Field(description="The reason behind choosing the particular category")


supervisor_prompt= read_prompt(os.environ["SUPERVISOR_PROMPT_DIR"])

# Validating the output
parser = JsonOutputParser(pydantic_object= SupervisorOutput)

prompt = ChatPromptTemplate(
    messages= [
        ("system", supervisor_prompt),
        ("human", "{user_question}")
    ],
    input_variables= ["user_question"],
    partial_variables= {"format_instructions": parser.get_format_instructions()}
)

# Defining the pipeline
supervisor_chain_pipeline = (
    {"user_question": RunnablePassthrough()}
    |
    prompt
    |
    llm
    |
    parser
)






