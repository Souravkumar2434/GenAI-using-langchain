from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

st.title("Fun chatting with OpenAI LLM")

st.write("This is a simple chat interface between user and OPEnAI's LLM.")
st.write("You can ask any question and get a response from the LLM.")

user_input = st.text_input("Enter you text here:", key="user_input")

if st.button("Ask"):
    if user_input:
        # Initialize the OpenAI LLM with the specified model and temperature
        llm = ChatOpenAI(model="gpt-4", temperature=0.7)
        
        # Call the LLM with the user input
        result = llm.invoke(user_input)
        
        st.write("Response from LLM:")
        st.write(result.content)
    else:
        st.warning("Please enter a message before sending.")