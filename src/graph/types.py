from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

from src.prompt.planner_model import Plan


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    locale: str = "zh-CN"
    messages: Annotated[list, add_messages]
    current_plan: Plan | str = None
    plan_iterations: int = 0
