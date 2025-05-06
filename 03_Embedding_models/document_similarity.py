from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


load_dotenv()

model = OpenAIEmbeddings(model="text-embedding-3-large", dimensions=300)

documents = [
    "Virat Kohli is known for his aggressive batting and consistent run-scoring across all formats.",
    "MS Dhoni is celebrated for his calm captaincy and finishing skills under pressure.",
    "Sachin Tendulkar is regarded as one of the greatest batsmen in the history of cricket.",
    "Ben Stokes is an all-rounder famous for his match-winning performances in crucial games.",
    "Steve Smith is known for his unorthodox technique and high Test batting average.",
    "Babar Azam is Pakistan's premier batsman, admired for his elegant strokeplay.",
    "Jasprit Bumrah is recognized for his deadly yorkers and pace variations in limited-overs cricket.",
    "Kane Williamson is respected for his composed batting and strategic captaincy.",
    "Rashid Khan is a world-class leg-spinner with exceptional control and variations.",
    "Joe Root is a technically sound batsman and a mainstay in England’s Test lineup."
]

query = "Who is MS Dhoni?"

# Embedding of documents
documents_vector = model.embed_documents(documents)

# Embedding of the query
query_vector = model.embed_query(query)


# Calculate cosine similarity between the query and each document
similarities = cosine_similarity([query_vector], documents_vector)[0]

print("similarity scores:", similarities)
print("Most similar document:", documents[np.argmax(similarities)])