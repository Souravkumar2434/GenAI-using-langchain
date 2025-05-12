from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

from dotenv import load_dotenv

load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.7,
        max_new_tokens=512,
    )

)


model1 = ChatHuggingFace(llm = llm)

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    input_variables=["input"],
    template = "Generate a detailed report on {input}."
)

prompt2 = PromptTemplate(
    input_variables=["text"],
    template = "Summarize the report on {text} in 5 points."
)

parser = StrOutputParser()

chain = prompt1 | model2 | parser | prompt2 | model1 | parser
result = chain.invoke({"input": "Python programming language"})

print(result)
chain.get_graph().print_ascii()