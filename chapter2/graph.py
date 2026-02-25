import os

import requests
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_classic import hub
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from pydantic import Field, BaseModel

load_dotenv(override=True)


class WeatherQuery(BaseModel):
    loc: str = Field(description="城市名称")


@tool(args_schema=WeatherQuery)
def get_weather(loc):
    """
        查询即时天气函数
        :param loc: 必要参数，字符串类型，用于表示查询天气的具体城市名称，\
        :return：心知天气 API查询即时天气的结果，具体URL请求地址为："https://api.seniverse.com/v3/weather/now.json"
        返回结果对象类型为解析之后的JSON格式对象，并用字符串形式进行表示，其中包含了全部重要的天气信息
    """
    url = "https://api.seniverse.com/v3/weather/now.json"
    params = {
        "key": "天气API_KEY",
        "location": loc,
        "language": "zh-Hans",
        "unit": "c",
    }
    response = requests.get(url, params=params)
    temperature = response.json()
    return temperature['results'][0]['now']


tools = [get_weather]

# 创建模型
model = init_chat_model(model="Qwen/Qwen3-8B",
                     model_provider="openai",
                     base_url="https://api.siliconflow.cn/v1/",
                     api_key=os.getenv("API_KEY")
                     )

prompt = hub.pull("hwchase17/react")

# 创建图
graph = create_react_agent(model=model, tools=tools)
