from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st



load_dotenv()

# Initialize the OpenAI LLM with the specified model and temperature
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

st.title("Chat with OpenAI LLM")
st.write("This is a simple chat interface using Streamlit and OpenAI's LLM.")

user_input = st.text_input("Enter your message:", key="user_input")
if st.button("Summarize"):
    if user_input:
        # Call the LLM with the user input
        result = llm.invoke(user_input)
        st.write("Response from LLM:")
        st.write(result.content)
    else:
        st.warning("Please enter a message before sending.")