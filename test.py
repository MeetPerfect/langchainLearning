import json

from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from pydantic import Field, BaseModel

# state = {"tra": '{"combat_scenario": "红-01 vs 蓝-A 空间对抗", "red_01": {"initial_position": [50, 50], "final_position": [54.178140325395646, 55.114728815232425], "initial_velocity": [5, 5], "final_velocity": [4.25498633985492, 636198631670363]}, "blue_a": {"initial_position": [60, 60], "final_position": [59.145228585452, 59.4288568051127], "initial_velocity": [-1, -1], "final_velocity": [-0.9154956451869662, -0.2687241018228503]}, "hvts": {"red_hvt": [1, 1], "blue_hvt": [0, 0]}}', "a": '{1: 1}'}
#
# for e in state:
#     try:
#         state[e] = json.loads(state[e])
#
#     except:
#         pass
# print(json.dumps(state, ensure_ascii=False, indent=4))
# print(type(state))
# print(state)


# trajectory_json = {"红-01": [[0,0,0], [1,1,0], [2,2,0]], "蓝-A": [[5,5,0], [4,4,0], [3,3,0]]}
# print(trajectory_json, type(trajectory_json))
#
# str = json.dumps(trajectory_json, ensure_ascii=False)
# print(str, type(str))
# data = json.loads(str)
#
# print(data, type(data))


schemas = [
    ResponseSchema(name="name", description="用户姓名"),
    ResponseSchema(name="age", description="用户的年龄")
]

parser = StructuredOutputParser.from_response_schemas(schemas)

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-lzcunbanmnmklpxtfehbnmupbgytyqgujjulndjtvhzjhqdq",
)

prompt = ChatPromptTemplate.from_template(
    "请根据以下内容提取用户信息，并返回 JSON 格式：\n{input}\n\n{format_instructions}"
)

chain = (
        prompt.partial(format_instructions=parser.get_format_instructions())
        | model
        | parser
)

result = chain.invoke({"input": "用户叫李雷，今年25岁，是一名工程师。"})  # 输入input, format_instructions前面已经赋值
print(result)
