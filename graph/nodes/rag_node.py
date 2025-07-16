from memory.state import State
from tools.rag_hybrid_retriever import hybrid_search_rag_pipeline


def rag_node(state:State):
    
    question = state['messages'][0]
    # Always extract .content if present
    question = getattr(question, 'content', question)

    print("\nHybrid search RAG call initiated")

    response = hybrid_search_rag_pipeline.invoke(question)

    print("\nRag Call's answer to the query: ", response['output'])

    # Adding the output to the state's messages
    return {"messages": [response['output']]}


