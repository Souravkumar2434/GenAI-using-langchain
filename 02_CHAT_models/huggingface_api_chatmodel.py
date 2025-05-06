from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation")

# Initialize the Hugging Face LLM with the specified model and temperature
model = ChatHuggingFace(llm =llm, temperature=0.7)

result = model.invoke("Who is the prime minister of India?")
print(result.content)