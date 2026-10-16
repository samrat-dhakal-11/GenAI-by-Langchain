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
    temperature=0.7,
    max_tokens=1000
)

paper_input = st.selectbox( "Select Research Paper Name", ["Attention Is All You Need", "BERT: Pre-training of Deep Bidirectional Transformers", "GPT-3: Language Models are Few-Shot Learners", "Diffusion Models Beat GANs on Image Synthesis"] )

style_input = st.selectbox( "Select Explanation Style", ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] ) 

length_input = st.selectbox( "Select Explanation Length", ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"] )


template=PromptTemplate(
  template = """
Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanation Style: {style_input}  
Explanation Length: {length_input}  

1. Mathematical Details:  
   - Include relevant mathematical equations if present in the paper.  
   - Explain the mathematical concepts using simple, intuitive code snippets where applicable.  
2. Analogies:  
   - Use relatable analogies to simplify complex ideas.  

If certain information is not available in the paper, respond with: "Insufficient information available" instead of guessing.  
Ensure the summary is clear, accurate, and aligned with the provided style and length.
""",
input_variables=['paper_input','style_input','length_input']
)

prompt=template.invoke({
  'paper_input':paper_input,
  'style_input':style_input,
  'length_input':length_input
})
st.title('SAMRATTTT')
st.header('Research Tool')

if st.button('Summarize'):
  result=llm.invoke(prompt)
  st.write(result.content)