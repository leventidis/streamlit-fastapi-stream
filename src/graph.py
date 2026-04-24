from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, MessagesState, START, END


def chatbot(state: MessagesState):
    """Chatbot function that takes the current state and returns the next state with the LLM response."""
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        streaming=True,
        temperature=0.0
    )
    return {
        "messages": [llm.invoke(state["messages"])]
    }


def build_graph():
    """Builds the LangGraph state graph for the chatbot."""
    graph_builder = StateGraph(MessagesState)
    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    
    return graph