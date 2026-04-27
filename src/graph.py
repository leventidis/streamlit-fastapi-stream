from typing import TypedDict
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END


class GraphState(TypedDict):
    question: str
    plan: str
    joke: str
    answer: str
    planning_required: bool


def orchestrate(state: GraphState):
    """If the questions is """
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.0)
    messages = [
        SystemMessage(
            content=(
                "You are a routing assistant. Your job is to decide whether a question requires structured planning before answering.\n\n"
                "Reply with 'planning' if the question:\n"
                "- Requires multiple steps or a sequence of actions to answer well\n"
                "- Is a how-to, tutorial, or process-oriented question\n"
                "- Involves research, comparison, or analysis across multiple areas\n"
                "- Would benefit from a structured, organized response\n\n"
                "Reply with 'answer_question' if the question:\n"
                "- Is a simple factual lookup (e.g. 'What is the capital of France?')\n"
                "- Is a greeting or small talk (e.g. 'hi', 'how are you')\n"
                "- Can be fully answered in one or two sentences\n\n"
                "Reply with only 'planning' or 'answer_question'. Do not include any other text."
            )
        ),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages).content.strip().lower()
    
    return {"planning_required": response == "planning"}


def should_plan(state: GraphState):
    if state["planning_required"]:
        return "planning"
    
    return "answer_question"


def planning(state: GraphState):
    """Given a question, come up with a plan to answer it."""
    llm = ChatOpenAI(model="gpt-4o-mini", streaming=True, temperature=0.0)
    messages = [
        SystemMessage(content="You are a helpful assistant. Given a question, create a concise step-by-step plan to answer it. Use at most 5 steps."),
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

    system_prompt = "You are a helpful assistant. Your task is to answer the user's question as best as you can."
    if plan := state.get("plan"):
        system_prompt += f"\n\nUse the following plan to structure your answer:\n{plan}"
    if joke := state.get("joke"):
        system_prompt += f"\n\nInclude the following joke in your answer to make it more engaging:\n{joke}"

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=state["question"]),
    ]
    response = llm.invoke(messages)
    return {"answer": response.content}


def build_graph():
    """Builds the LangGraph state graph with planning, joke, and answer nodes."""
    graph_builder = StateGraph(GraphState)

    graph_builder.add_node("orchestrate", orchestrate)
    graph_builder.add_node("planning", planning)
    graph_builder.add_node("generate_joke", generate_joke)
    graph_builder.add_node("answer_question", answer_question)

    graph_builder.add_edge(START, "orchestrate")
    graph_builder.add_conditional_edges("orchestrate", should_plan)
    graph_builder.add_edge("planning", "generate_joke")
    graph_builder.add_edge("generate_joke", "answer_question")
    graph_builder.add_edge("answer_question", END)

    return graph_builder.compile()