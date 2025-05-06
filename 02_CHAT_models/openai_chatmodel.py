from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4", temperature=0.7, max_completion_tokens = 10)

result = llm.invoke("Who is the captain of Indian cricket team?")

print(result.content)

