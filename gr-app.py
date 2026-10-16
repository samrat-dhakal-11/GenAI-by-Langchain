print("hello")
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
print("stage-1")
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.5,
    max_tokens=100
)

print("stage-2")
text=input("Enter your qsn")
print("stage-3")
result=llm.invoke(text)
print("stage-4")
print(result.content)