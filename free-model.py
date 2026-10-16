print("helloo")
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()
print("stage-1")
llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    temperature=0.5,
    max_new_tokens=200,
   # timeout=60
)
print("stage-2")
model = ChatHuggingFace(llm=llm)
print("stage-3")
result = model.invoke("What the capital of Nepal ")
print(result.content)

