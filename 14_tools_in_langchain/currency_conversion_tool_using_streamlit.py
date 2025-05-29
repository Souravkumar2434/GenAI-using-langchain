import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.tools import InjectedToolArg
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests
import json
from typing import Annotated

st.title("Currency Conversion Tool")



load_dotenv()

@tool
def get_conversion_factor(from_currency: str, to_currency: str) -> str:
    """
    Get the conversion factor between two currencies.
    """
    url = f"https://v6.exchangerate-api.com/v6/91b32d2b034022793a835278/pair/{from_currency}/{to_currency}"
    response = requests.get(url)
    
    if response.status_code != 200:
        raise ValueError(f"Error fetching conversion rate: {response.text}")
    
    data = json.loads(response.text)
    return data['conversion_rate']

@tool
def convert_currency(amount: int, coversion_factor: Annotated[float, InjectedToolArg]) -> float:
    """
    Convert an amount of money using the provided conversion factor.
    """
    return amount * float(coversion_factor)


llm = ChatOpenAI()

llm_with_tools = llm.bind_tools([get_conversion_factor, convert_currency])

from_currency = st.text_input("From Currency (e.g., USD):")
to_currency = st.text_input("To Currency (e.g., EUR):")
amount = st.number_input("Amount to Convert:", min_value=1, step=1)

messages = []

messages.append(HumanMessage(content=f"Convert {amount} {from_currency} to {to_currency} and provide the conversion rate also"))


if st.button("Convert"):
    ai_message = llm_with_tools.invoke(messages)
    messages.append(ai_message)
    for tool_call in ai_message.tool_calls:
        if tool_call['name'] == "get_conversion_factor":
            conversion_factor = get_conversion_factor.invoke(tool_call)
            print(f"Conversion factor: {conversion_factor}")
            messages.append(conversion_factor)
        if tool_call['name'] == "convert_currency":
            tool_call['args']['coversion_factor'] = conversion_factor.content
            converted_amount = convert_currency.invoke(tool_call)
            print(f"Converted amount: {converted_amount}")
            messages.append(converted_amount)
    result = llm_with_tools.invoke(messages)

    st.write("Conversion Result:")
    st.write(result.content)