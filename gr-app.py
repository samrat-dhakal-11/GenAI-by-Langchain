from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()


llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=100
)

text=input("Enter your qsn")

result=llm.invoke(text)

print(result.content)
