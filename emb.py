import operator
from typing import TypedDict, Annotated
from langgraph.types import Send
from langgraph.graph import START, END, StateGraph

# 1. Define the state schema
class SubSectionState(TypedDict):
    topic: str
    section_name: str

class BlogState(TypedDict):
    main_topic: str
    sections: list[str]  # Sub-topics to generate
    # operator.add acts as the reducer to combine parallel outputs
    blog_content: Annotated[list[str], operator.add] 

# 2. Define the Router / Conditional Edge using Send
def fan_out_sections(state: BlogState):
    """
    Dynamically creates a Send object for each section in state['sections'].
    This routes execution to the 'generate_section' node multiple times in parallel.
    """
    print(f"\n--- [ROUTER] fan_out_sections triggered ---")
    
    sends = []
    for section in state["sections"]:
        # Send target node name AND custom isolated state (arg)
        packet = Send("generate_section", {
            "topic": state["main_topic"], 
            "section_name": section
        })
        sends.append(packet)
        print(f"👉 Generated Send packet: node='generate_section', arg={{'topic': '{state['main_topic']}', 'section_name': '{section}'}}")
        
    return sends

# 3. Define the Worker Node
def generate_section(state: SubSectionState):
    """
    This worker node runs in parallel for each Send packet.
    It receives the custom 'SubSectionState' (not the full BlogState).
    """
    topic = state["topic"]
    section = state["section_name"]
    
    print(f"   ⚡ [WORKER NODE] Executing generate_section for: '{section}'")
    
    # Simulate generating content for that specific section
    generated_text = f"## {section}\nThis is the generated content covering {section} of {topic}.\n"
    
    # Return the dictionary matching the reducer field in BlogState
    return {"blog_content": [generated_text]}

# 4. Build the Graph
builder = StateGraph(BlogState)

# Add nodes
builder.add_node("generate_section", generate_section)

# Add conditional edges from START using our fan-out router
builder.add_conditional_edges(START, fan_out_sections)

# Connect worker output to END
builder.add_edge("generate_section", END)

# Compile graph
graph = builder.compile()

# 5. Invoke the Graph
initial_input = {
    "main_topic": "LangGraph",
    "sections": ["Introduction", "Why Use Send?", "Conclusion"]
}

print("=== STARTING GRAPH INVOCATION ===")
final_output = graph.invoke(initial_input)

print("\n=== FINAL COMPILED OUTPUT ===")
print("\n".join(final_output["blog_content"]))
