from memory.state import State
from pydantic import BaseModel, Field
from tools.supervisor_respond import supervisor_chain_pipeline

class SupervisorInput(BaseModel):
    user_question: str= Field(description="This is the user's question, that needs to be categorized")


def router_supervisor_node(state:State):
    question = state['messages'][-1]
    # Always extract .content if present
    question = getattr(question, 'content', question)

    validated_user_query = SupervisorInput(user_question=question)
    validated_question = validated_user_query.user_question

    print("\nSupervising for routing")
    print("\nQuestion: ", validated_question)

    # Invoking the pipeline, with data validation- pydantic
    response = supervisor_chain_pipeline.invoke(validated_question)

    print("\nSupervisor Decision: ", response['category'])
    print("\nReason: ", response['reason'])

    # Adding the category to the state's messages
    return {"messages": [response['category']]}


