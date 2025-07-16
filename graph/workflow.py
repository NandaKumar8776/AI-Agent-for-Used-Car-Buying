from langgraph.graph import StateGraph,END
from memory.state import State

from graph.nodes.supervisor_node import router_supervisor_node
from graph.nodes.llm_node import llm_node
from graph.nodes.rag_node import rag_node
from graph.nodes.web_crawler_node import web_crawler_node


workflow = StateGraph(State)

# Router (Handle Supervisor's response and divert accordingly)

def router(state:State):
    print("-> ROUTER ->")
    
    route = state["messages"][-1]
    # Always extract .content if present
    route = getattr(route, 'content', route)

    if "used car" in route.lower():
        return "Used Car Call"
    elif "car dealership inventory website" in route.lower():
        return "Car Inventory Call"
    else:
        return "LLM Call"

# Adding node functions to the workflow (Each with data validation -in and -out, hybrid search as needed)

workflow.add_node("Supervisor", router_supervisor_node)

workflow.add_node("LLM_call", llm_node)

workflow.add_node("RAG_call", rag_node)

workflow.add_node("Crawler_call", web_crawler_node)

workflow.set_entry_point("Supervisor")

# Adding the conditional edges (Supervisor -> router -> LLM, or RAG, or Web Scraper RAG)

workflow.add_conditional_edges(
    "Supervisor",
    router,
    {
        "Used Car Call": "RAG_call",
        "Car Inventory Call": "Crawler_call",
        "LLM Call": "LLM_call",
    }
)

# Adding the edges (All -> END)

workflow.add_edge("RAG_call",END)
workflow.add_edge("Crawler_call",END)
workflow.add_edge("LLM_call",END)

# Compiling the LangGraph app

app= workflow.compile()







