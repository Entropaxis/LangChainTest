import time
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import AgentMiddleware, AgentState
from langchain.messages import SystemMessage, HumanMessage, AIMessage


load_dotenv()

class CustomMiddleware(AgentMiddleware):

    def __init__(self):
        super().__init__()
        self.star_time = 0.0

    def before_agent(self, state, runtime):
        self.star_time = time.time()
        print("before agent triggered")

    def before_model(self, state, runtime):
        print("before model triggered")

    def after_model(self, state, runtime):
        print("after model triggered")

    def after_agent(self, state, runtime):
        print("after_agent:", time.time() - self.star_time)

agent = create_agent(
    model="gpt-5.4-mini",
    middleware=[CustomMiddleware()]
)

response = agent.invoke({
    'messages': [
        SystemMessage('You are a helpful assistant.'),
        HumanMessage('What is 1 + 1?'),
    ]
    
})

print(response['messages'][-1].content)