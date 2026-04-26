from dataclasses import dataclass

import requests
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()


@dataclass
class Context:
    user_id: str

@dataclass
class ResponseFormat:
    summary: str
    temperature_celsius: float
    temperature_farenheit: float
    humidity: float


@tool("get_weather", description="Get the weather for a location")
def get_weather(location: str) -> str:
    response = requests.get(f'https://wttr.in/{location}?format=j1')
    return response.json()


@tool("locate_user", description = "Look up a users's city based on the context")
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case "ABC123":
            return "New York"
        case "DEF456":
            return "San Francisco"
        case "GHI789":
            return "Los Angeles"
        case _:
            return "Unknown"

model = init_chat_model(model="gpt-5.4-nano", temperature=0.3)


checkpointer = InMemorySaver()

agent = create_agent(
    tools=[get_weather, locate_user],
    model= model,
    system_prompt="You are a helpful assistant.",
    context_schema=Context,
    response_format=ResponseFormat,
    checkpointer=checkpointer,
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    "messages": [
        {"role": "user", "content": "What is the weather like"}
        ]},
        config = config,
        context=Context(user_id="DEF456")
)

print(response['structured_response'])
print(response['structured_response'].summary)
print(response['structured_response'].temperature_celsius)


