from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    input_variables=["input"],
    template = "Generate a list of 5 interesting facts about {input}."
)

model = ChatOpenAI()

parser = StrOutputParser()

chain = prompt | model | parser


result = chain.invoke({"input": "Python programming language"})
print(result)