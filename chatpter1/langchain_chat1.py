

# description: 使用 LangChain 构建一个简单的聊天机器人单论对话，模拟日本著名女演员苍井空的对话风格。
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", "你叫苍井空，是日本著名女演员。"),
    ("human", "{input}")
])

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key="sk-lzcunbanmnmklpxtfehbnmupbgytyqgujjulndjtvhzjhqdq",
)

basic_chain = chatbot_prompt | model | StrOutputParser()

question = "你好，请你介绍一下你自己。"
result = basic_chain.invoke(question)
print(result)