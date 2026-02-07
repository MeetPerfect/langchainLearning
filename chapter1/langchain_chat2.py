import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# description: langchain添加多轮对话记忆功能

prompt = ChatPromptTemplate.from_messages([
    SystemMessage(content="你叫苍井空，是日本著名女演员。"),
    MessagesPlaceholder(variable_name="messages"),
])

load_dotenv()

model = init_chat_model(
    model="Qwen/Qwen3-8B",
    model_provider="openai",
    base_url="https://api.siliconflow.cn/v1/",
    api_key=os.getenv("API_KEY"),
)

chain = prompt | model | StrOutputParser()

messages_list = []
print("🔹 输入 exit 结束对话")

while True:
    user_query = input("用户: ")

    if user_query.lower() in {"exit", "quit"}:
        print("对话结束。")
        break
    messages_list.append(HumanMessage(content=user_query))

    response = chain.invoke({"messages": messages_list})

    print("苍老师: ", response)

    messages_list.append(AIMessage(content=response))

    messages_list = messages_list[-50:]
