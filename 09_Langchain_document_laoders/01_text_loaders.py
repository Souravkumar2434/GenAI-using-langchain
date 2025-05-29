from langchain_community.document_loaders import TextLoader

loader = TextLoader("cricket_summary.txt", encoding="utf-8")
data = loader.load()
print("--------------------------------------------------")
print(data[0].metadata)
print("--------------------------------------------------")
print(data[0].page_content)


print("--------------------------------------------------")
print(type(data))
print("--------------------------------------------------")
print(type(data[0]))