import os

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# description: langchain工具调用示例，集成天气查询API

load_dotenv()


@tool
def get_weather(location: str):
    """
        查询即时天气函数
        :param location: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "心知API",
        "location": location,
        "language": "zh-Hans",
        "unit": "c"
    }

    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature["results"][0]["now"]


prompt = ChatPromptTemplate.from_messages([
    ("system", "你是天气助手，请根据用户的问题，给出相应的天气信息能力"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

tools = [get_weather]

agent_runnable = create_tool_calling_agent(model, tools, prompt)
agent = AgentExecutor(agent=agent_runnable, tools=tools, verbose=True)
response = agent.invoke({"input": "请问今天北京天气怎么样？"})
print(response)
