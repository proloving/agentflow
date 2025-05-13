from langgraph.graph import StateGraph, START, END
from .types import State
from .node import (
    coordinator_node,
    planner_node,
    research_team_node,
    human_feedback_node,
    researcher_node,
    reporter_node,
)


def _build_base_graph():
    """Build and return the base state graph with all nodes and edges."""
    builder = StateGraph(State)
    builder.add_edge(START, "coordinator")
    builder.add_node("coordinator", coordinator_node)
    builder.add_node("planner", planner_node)
    builder.add_node("human_feedback", human_feedback_node)
    builder.add_node("research_team", research_team_node)
    builder.add_node("researcher", researcher_node)
    builder.add_node("reporter", reporter_node)
    builder.add_edge("reporter", END)
    return builder


def build_graph():
    """Build and return the agent workflow graph without memory."""
    # build state graph
    builder = _build_base_graph()
    return builder.compile()
