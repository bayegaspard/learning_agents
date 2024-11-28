from langchain_ollama import ChatOllama  # Use ChatOllama from langchain-ollama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from typing import List

# Define the tool to search for weather
@tool
def search(query: str):
    """Search for weather or general information."""
    print(f"Search tool called with query: {query}")  # Debugging statement
    # Placeholder logic for search tool
    if "sf" in query.lower() or "san francisco" in query.lower():
        return "It's 60 degrees and foggy."
    return "It's 90 degrees and sunny."


# Define the tools that the model will use
tools = [
    {
        "name": "search_weather",
        "description": "Search for weather in a specific city.",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g., San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "The unit for temperature.",
                },
            },
            "required": ["location"],
        },
    }
]

# Define a custom prompt template for the tool system (just an example)
toolSystemPromptTemplate = "You are a helpful assistant that can answer questions using tools. You can use the search tool for weather queries."

# Initialize the ChatOllama model with tool bindings
model = ChatOllama(
    model="artifish/llama3.2-uncensored",  # Example model, replace as needed
    toolSystemPromptTemplate=toolSystemPromptTemplate,
).bind(
    functions=tools,
    function_call={"name": "search_weather"}
)

# Example function to call the model with user input and get the response
def call_model_with_tool(user_query: str):
    # Create the user message
    user_message = HumanMessage(content=user_query)

    # Invoke the model with the user query
    response = model.invoke([user_message])

    # Check if there is a valid response and output it
    if response:
        # The response is an AIMessage object, directly access the content attribute
        ai_message = response
        print(f"Model response: {ai_message.content}")  # Access the content directly
    else:
        print("No response from model.")
    
    return ai_message.content if ai_message else None


# Example usage of the model with the weather query
user_query = "What's the weather in San Francisco?"
call_model_with_tool(user_query)

# Example usage of the model with another query that doesn't trigger a tool
user_query_2 = "Tell me a joke."
call_model_with_tool(user_query_2)
