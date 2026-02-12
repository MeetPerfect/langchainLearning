import os

import dotenv
import requests
from langchain.chat_models import init_chat_model
from langchain_classic.agents import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from pydantic import BaseModel, Field

# 多工具调用

dotenv.load_dotenv()


class WeatherQuery(BaseModel):
    location: str = Field(description="城市")


class WriteQuery(BaseModel):
    content: str = Field(description="需要写入文档的具体内容")


@tool(args_schema=WeatherQuery)
def get_weather(location):
    """
            查询即时天气函数
            :param location: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
            :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
            返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
        """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "你注册的心知天气私钥",
        "location": location,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']


print(f"name: {get_weather.name}, description: {get_weather.description}, arguments: {get_weather.args}")


@tool(args_schema=WriteQuery)
def write_file(content):
    """
    将指定内容写入文件
    :param content:
    :return:
    """
    with open("test.txt", "w", encoding="utf-8") as f:
        f.write(content)
    return "已成功写入本地文件。"

prompt = ChatPromptTemplate(

)

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

tools = [get_weather, write_file]

agent = create_react_agent(model, tools)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "北京和杭州现在的天气如何?并把查询结果写入文件中"
            }
        ]
    }
)

print(response)
