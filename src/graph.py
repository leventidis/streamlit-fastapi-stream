from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    plan: str
    joke: str
    answer: str


def planning(state: GraphState):
    """Given a question, come up with a plan to answer it."""
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.0)
    messages = [
        SystemMessage(content="You are a helpful assistant. Given a question, create a concise step-by-step plan to answer it."),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {"plan": response.content}


def generate_joke(state: GraphState):
    """Generates a concise joke related to the question topic."""
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.7)
    messages = [
        SystemMessage(content="You are a comedian. Generate a single short, funny joke related to the topic of the question. Keep it to 1-2 sentences."),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {"joke": response.content}


def answer_question(state: GraphState):
    """Generates the final answer to the question using the plan."""
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.0)
    messages = [
        SystemMessage(content=f"You are a helpful assistant. Use the following plan to answer the question.\n\nPlan:\n{state['plan']}. Include the following joke in your answer to make it more engaging:\n\nJoke:\n{state['joke']}"),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph():
    """Builds the LangGraph state graph with planning, joke, and answer nodes."""
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("planning", planning)
    graph_builder.add_node("generate_joke", generate_joke)
    graph_builder.add_node("answer_question", answer_question)

    graph_builder.add_edge(START, "planning")
    graph_builder.add_edge("planning", "generate_joke")
    graph_builder.add_edge("generate_joke", "answer_question")
    graph_builder.add_edge("answer_question", END)

    return graph_builder.compile()