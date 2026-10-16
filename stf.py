import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
import streamlit as st
load_dotenv(dotenv_path="/public/Samrat_Learning_Gen_Ai/.env")
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=0.7,
    max_tokens=1000
)
st.title('SAMRATTTT')
st.header('Research Tool')
user_input=st.text_input('Enter your Prompt')

if st.button('Summarize'):
  result=llm.invoke(user_input)
  st.write(result.content)