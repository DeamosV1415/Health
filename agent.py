"""
Health Chatbot Agent
Contains the LangGraph agent logic, tools, and workflow.
"""

import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain.tools import tool
from pydantic import BaseModel, Field
from langgraph.prebuilt import ToolNode
from langchain_tavily import TavilySearch

# Load environment variables
load_dotenv(override=True)


# State definition
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# Tool schema
class GeneralSearch(BaseModel):
    """Input for the general search tool"""
    query: str = Field(default="", description="Search query")


# Define the search tool
@tool(args_schema=GeneralSearch)
def general_search(query: str):
    """General search tool"""
    general_search_tool = TavilySearch(
        max_results=5,
        topic="general"
    )
    return general_search_tool.invoke(query)


# System message for the health chatbot
SYSTEM_MESSAGE = """You are a medical information assistant. 
When a user asks a health question:
1. Use the general_search tool to find accurate medical information
2. After receiving search results, provide a clear, helpful answer
3. Include appropriate medical disclaimers when needed.
4. Always be sure to explain as if the user is in 5th grade. If some complex medical terms come up, be sure to simplify them. Don't use too much medical jargon.
Important: 
- Respond naturally in plain text, not JSON
- After you get tool results, synthesize them into a helpful answer
- Always inclue some relevant medical questions for your doctor about the query.
- For every query, when you give you verdict, always give a coloured alert. 
- Give a coloured alert to the user. 🟢 Green if the user is safe, 🟡 Yellow if the user is at risk, 🟠 Orange if the user is at high risk, 🔴 Red if the user is at very high risk.
- Following the colour alert, add an advise to the user.
After this continue with all the other important steps.
""".strip()


# Initialize LLM with tools
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=1.0,  # Gemini 3.0+ defaults to 1.0
    max_tokens=None,
    timeout=None,
    max_retries=2
)
llm_with_tools = llm.bind_tools([general_search])


def chatbot_node(state: AgentState) -> AgentState:
    """Process messages and decide if tools are needed"""
    
    found_system_message = False
    messages = state["messages"]
    for message in messages:
        if isinstance(message, SystemMessage):
            message.content = SYSTEM_MESSAGE
            found_system_message = True

    if not found_system_message:
        messages = [SystemMessage(content=SYSTEM_MESSAGE)] + messages

    response = llm_with_tools.invoke(messages)
    
    return {
        "messages": [response],
    }
    

def should_continue(state: AgentState):
    """Check if we need to call tools"""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# Build the workflow
workflow = StateGraph(AgentState)

workflow.add_node("chatbot", chatbot_node)
workflow.add_node("tools", ToolNode(tools=[general_search]))

workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
workflow.add_edge("tools", "chatbot")

# Compile the graph with memory
checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)


def get_chatbot_response(user_message: str, thread_id: str = "1") -> str:
    """
    Get a response from the health chatbot.
    
    Args:
        user_message: The user's input message
        thread_id: Conversation thread ID for maintaining context
        
    Returns:
        The chatbot's response as a string
    """
    config = {"configurable": {"thread_id": thread_id}}
    
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_message)]}, 
        config=config
    )
    
    return result["messages"][-1].content


if __name__ == "__main__":
    # Test the agent
    print("Health Chatbot Agent - Test Mode")
    print("-" * 50)
    
    response = get_chatbot_response("What are the symptoms of flu?")
    print(f"Response: {response}")
