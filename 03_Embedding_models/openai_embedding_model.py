from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

# Initialize the OpenAI Embedding model with the specified model and temperature
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=32)

documents = [
    "The prime minister of India is Narendra Modi.",
    "The capital of France is Paris.",
    "The largest ocean on Earth is the Pacific Ocean.",
]
# To run for a single sentence

# model = embedding_model.embed_query("Who is the prime minister of India?")

# to run for several sentences using documents
model = embedding_model.embed_documents(documents)
print(str(model))