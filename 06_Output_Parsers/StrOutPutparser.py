from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate


load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="ArcherH/naturallama-beta-v0.01",
    task = "text-generation",
)

model = ChatHuggingFace(llm = llm)

template1 = PromptTemplate(input_variables=["input"], template="provide a detailed report on the given topic: {input}")
template2 = PromptTemplate(input_variables=["text"], template="provide a summary in 5-6 lines of the given topic: {text}")

parser = StrOutputParser()

chain = template1| model | parser | template2 | model | parser

result = chain.invoke({"input": "The impact of climate change on global agriculture"})

print(result)
