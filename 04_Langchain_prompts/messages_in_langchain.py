from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

# Create a system message to set the context for the conversation
system_message = SystemMessage(content="You are a helpful assistant.")
human_message = HumanMessage(content="Who is the captain of Indian cricket team?")

messages = [system_message, human_message]
result = model.invoke(messages)
print(result.content)

# Append the AI's response to the messages list
messages.append(AIMessage(content=result.content))

print(messages)