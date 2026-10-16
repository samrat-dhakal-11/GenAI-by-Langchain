import os 
import streamlit as st
import questionary
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate,load_prompt
from dotenv import load_dotenv

load_dotenv('/public/Samrat_Learning_Gen_Ai/.env')
api_key= os.getenv('GROQ_API_KEY')

modell=ChatGroq(
  model="openai/gpt-oss-20b",
  api_key=api_key
)
template=load_prompt("/public/Samrat_Learning_Gen_Ai/template2.json")

print("="*48)
print("="*48)

user_name=input("Hi User🤙,Enter your Name: ")
print("-"*48)

go_chat=True

def prompt_style():
  print("="*48)
  print()
  style_input=questionary.select(
        f"Select Your Output Style {user_name}😁",
        ["Beginner-Friendly", "Technical", "Code-Oriented", "Mathematical"] 
        ).ask()
  
  print()
  length_input=questionary.select(
        f"Select Your Length Style {user_name}😙",
        ["Short (1-2 paragraphs)", "Medium (3-5 paragraphs)", "Long (detailed explanation)"]
        ).ask()
  
  print("="*48)
  
  print(f"\nYou Choose Output Style:{style_input} \t Length Style:{length_input}")
  return style_input,length_input

print(f"\nBefore heading Do choose Output Style and Length{user_name}😘")
style_input,length_input=prompt_style()

print(f"\nIf you Wanna End Chat Just Write 'End'🙏 and If Style Change Just Write 'Changestyle'😊")
print()

print("-"*48)

print(f"🗣️ Assistant:Hello {user_name},Ask me a question")
print()

def chat_section(go_chat,style_input,length_input):
  while(go_chat):
    print("-"*48)
    user_query=input(f"Enter your qsn  dear {user_name}😎:")
    print(user_query)
    if(user_query.lower()=="end"):
      print("\nAs You Have written End Chat is Ended....🎇🎇🎆🎇🎆🎇🎆")
      go_chat=False
      
    elif(user_query.lower()=="changestyle"):
      style_input,length_input=prompt_style()
      
    else:
      chain= template | modell
      assistant_res=chain.invoke({
        'user_query':user_query,
        'style_input':style_input,
        'length_input':length_input
      })
      
      assistant_response=assistant_res.content
      print(f"\n🗣️ Assistant:\n{assistant_response}")
      print("-"*48)
      
  print(f"\nThanks For Interacting With Our Chat Model🎖️🎖️🎖️🎖️")

chat_section(go_chat,style_input,length_input)

def undo_function():
  print()
  undo_section=questionary.select(
    "Do You Wish To Go Back To Chat Section",
    ["Yes","No"]
  ).ask()

  return undo_section

undo=True
while(undo):
  undo_section=undo_function()
  if undo_section=="Yes":
    go_chat=True
    chat_section(go_chat,style_input,length_input)  
  else:
    undo=False
    print(f"\nThanks For Interacting With Our Chat Model🎖️🎖️🎖️🎖️")
#        python3 chatbot.py