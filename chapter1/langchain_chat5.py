import os

import gradio as gr
import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser, JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool

# description: langchain工具调用示例，集成天气查询API

load_dotenv()


@tool
def get_weather(location: str):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "心知天气api key",
        "location": location,
        "language": "zh-Hans",
        "unit": "c"
    }

    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature["results"][0]["now"]


model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

tools = [get_weather]

llm_with_tools = model.bind_tools(tools)

response = llm_with_tools.invoke("请帮我查询北京的天气情况？")
print(response)

parser = JsonOutputKeyToolsParser(key_name=get_weather.name, first_tool_only=True)


chain = llm_with_tools | parser

# response = chain.invoke("请帮我查询北京的天气情况？")
# print(response)

get_weather_chain = chain | get_weather

response = get_weather_chain.invoke("请帮我查询上海的天气情况？")

print(response)
