import config.env_setup
from graph.workflow import app


question = "What are the important things to consider for buying a used car with financing?"

state={"messages":[question]}

app.invoke(state)