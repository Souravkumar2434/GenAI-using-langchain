from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI()

st.title("Chat with an expert of your own interest about any topic")

domain = st.text_input("Enter your domain of interest:", key="domain")
topic = st.text_input("Enter your topic of interest:", key="topic")

st.write(f"You are talking to {domain} expert. You can ask any question related to {topic}.")

chat = ChatPromptTemplate([
    ('system', 'You are a {domain} expert'),
    ('human', "Explain about {topic} in detail")
])

prompt = chat.invoke({'domain':domain, 'topic':topic})

if st.button("Ask"):
    if domain and topic:
        # Call the LLM with the user input
        result = model.invoke(prompt)
        st.write("Response from the Expert:")
        st.write(result.content)
    else:
        st.warning("Please enter a message before sending.")