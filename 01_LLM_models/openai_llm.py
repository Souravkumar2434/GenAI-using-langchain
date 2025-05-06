from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

# Initialize the OpenAI LLM with the specified model and temperature
llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.7)

# The OpenAI LLM is now ready to be used for various tasks such as text generation, summarization, etc.
# You can use the `llm` object to call the OpenAI API and perform tasks with the specified model and temperature.
result = llm.invoke("Who is the prime minister of India?")

print(result)