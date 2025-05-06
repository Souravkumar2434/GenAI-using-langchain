from langchain_huggingface import HuggingFaceEmbeddings

llm = HuggingFaceEmbeddings(
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
)

# text = "The prime minister of India is Narendra Modi."
# vector = llm.embed_query(text)

documents = [
    "The prime minister of India is Narendra Modi.",
    "The capital of France is Paris.",
    "The largest ocean on Earth is the Pacific Ocean.",
]

vector = llm.embed_documents(documents)


print(vector)
print(len(vector), len(vector[0]))