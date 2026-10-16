import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate,load_prompt
import streamlit as st
load_dotenv(dotenv_path="/public/Samrat_Learning_Gen_Ai/.env")
api_key = os.getenv("GROQ_API_KEY")


llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-20b",
    temperature=1.7,
    max_tokens=5000
)
st.title('SAMRATTTT')
st.header('Research Tool')

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )

template=load_prompt('/public/Samrat_Learning_Gen_Ai/template.json')

if st.button('Summarize'):
  chain = template | llm
  result= chain.invoke({
    'paper_input':paper_input,
    'style_input':style_input,
    'length_input':length_input
  })
  st.write(result.content)