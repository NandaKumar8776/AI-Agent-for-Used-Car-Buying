from memory.state import State
from tools.web_crawler import hybrid_web_crawl_rag_pipeline


def web_crawler_node(state:State):
    
    # Validation already done at Supervisor node
    question = state['messages'][0]
    # Always extract .content if present
    question = getattr(question, 'content', question)

    print("\nWeb Crawler response generating")

    # Invoking the pipeline, with data validation- pydantic
    response = hybrid_web_crawl_rag_pipeline.invoke(question)

    print("\nGenerated response: ", response['output'])

    # Adding the category to the state's messages
    return {"messages": [response['output']]}


