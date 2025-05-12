from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
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

model1 = ChatHuggingFace(llm=llm)

model2 = ChatOpenAI()

prompt1 = PromptTemplate(
    input_variables=["input"],
    template="Prepare notes on {input}."
)

prompt2 = PromptTemplate(
    input_variables=["input"],
    template="Prepare a quiz of 2 question and answer based on {input}."
)

prompt3 = PromptTemplate(
    template = "Merges the notes and quiz into a single document.\n notes -> {notes}, quiz -> {quiz}",
    input_variables=["notes", "quiz"]
)

parser = StrOutputParser()

# Parallel chain
# The first two prompts are executed in parallel, and the third prompt is executed after both of them are completed.
# The output of the first two prompts is passed to the third prompt.
from langchain.schema.runnable import RunnableParallel

parallel_chain = RunnableParallel(
    {
        "notes": prompt1 | model1 | parser,
        "quiz": prompt2 | model2 | parser
    }
)
# The output of the parallel chain is passed to the third prompt.
final_chain = prompt3 | model1 | parser

chain = parallel_chain | final_chain

result = chain.invoke({"input": "Python is a versatile and widely-used programming language known for its simplicity and readability. Its clean syntax and dynamic typing make it an excellent choice for both beginners and experienced developers. Python supports multiple programming paradigms, including procedural, object-oriented, and functional programming. It has a vast standard library and an active community, providing extensive resources and tools for development. Popular for web development, data analysis, artificial intelligence, and automation, Python's flexibility has made it a go-to language in various industries. Its ease of use and wide applicability ensure its continued popularity in the programming world."})

print(result)