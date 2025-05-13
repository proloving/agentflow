import json
import logging

from src.config.configuration import Configuration
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from typing import Annotated, Literal
from langgraph.types import Command
from src.llms.llm import get_llm
from src.prompt.template import apply_prompt_template
from .types import State
from ..prompt.planner_model import StepType, Plan
from ..utils.json_utils import repair_json_output
from src.agent.agents import research_agent
from src.tools import (
    crawl_tool,
    web_search_tool,
)

logger = logging.getLogger(__name__)


@tool
def handoff_to_planner(
        task_title: Annotated[str, "The title of the task to be handed off."],
        locale: Annotated[str, "The user's detected language locale (e.g., en-US, zh-CN)."],
):
    """Handoff to planner agent to do plan."""
    # This tool is not returning anything: we're just using it
    # as a way for LLM to signal that it needs to hand off to planner agent
    logger.info("call handoff_to_planner")
    return


def coordinator_node(state: State) -> Command[Literal["planner", "__end__"]]:
    logger.info("Coordinator talking.")
    messages = apply_prompt_template("coordinator", state)
    response = (
        get_llm()
        .bind_tools([handoff_to_planner])
        .invoke(messages)
    )
    goto = "__end__"

    logger.info(f"Coordinator response: {response}")
    if len(response.tool_calls) > 0:
        goto = "planner"
        try:
            for tool_call in response.tool_calls:
                if tool_call.get("name", "") != "handoff_to_planner":
                    # todo
                    continue
                # if tool_locale := tool_call.get("args", {}).get("locale"):
                #     locale = tool_locale
                #     break
        except Exception as e:
            logger.error(f"Error processing tool calls: {e}")
    else:
        logger.warning(
            "Coordinator response contains no tool calls. Terminating workflow execution."
        )

    return Command(
        goto=goto,
    )


def planner_node(state: State, config: RunnableConfig) -> Command[Literal["reporter", "research_team", "__end__"]]:
    logger.info(f"Planner talking.state: {state}")
    configurable = Configuration.from_runnable_config(config)
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    if plan_iterations >= configurable.max_plan_iterations:
        logger.info("max plan iterations reached.")
        return Command(goto="reporter")

    messages = apply_prompt_template("planner", state)
    response = (
        get_llm().with_structured_output(
            Plan,
            method="json_mode",
        ).invoke(messages)
    )

    full_response = response.model_dump_json(indent=4, exclude_none=True)
    logger.info(f"Planner response: {full_response}")

    try:
        curr_plan = json.loads(repair_json_output(full_response))
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        return Command(goto="__end__")
        # if plan_iterations > 0:
        #     return Command(goto="reporter")
    if curr_plan.get("has_enough_context"):
        logger.info("Planner response has enough context.")
        new_plan = Plan.model_validate(curr_plan)
        return Command(
            update={
                "messages": [AIMessage(content=full_response, name="planner")],
                "current_plan": new_plan,
            },
            goto="reporter",
        )
    return Command(
        update={
            "messages": [AIMessage(content=full_response, name="planner")],
            "current_plan": full_response,
        },
        goto="human_feedback",
    )


def human_feedback_node(
        state,
) -> Command[Literal["planner", "research_team", "__end__"]]:
    logger.info(f"human feedback node. state: {state}")
    current_plan = state.get("current_plan", "")
    # if the plan is accepted, run the following node
    plan_iterations = state["plan_iterations"] if state.get("plan_iterations", 0) else 0
    goto = "research_team"
    try:
        current_plan = repair_json_output(current_plan)
        # increment the plan iterations
        plan_iterations += 1
        # parse the plan
        new_plan = json.loads(current_plan)
        if new_plan["has_enough_context"]:
            goto = "__end__"
    except json.JSONDecodeError:
        logger.warning("Planner response is not a valid JSON")
        if plan_iterations > 0:
            return Command(goto="reporter")
        else:
            return Command(goto="__end__")

    return Command(
        update={
            "current_plan": Plan.model_validate(new_plan),
            "plan_iterations": plan_iterations,
        },
        goto=goto,
    )


def research_team_node(
        state: State,
) -> Command[Literal["reporter", "researcher"]]:
    """Research team node that collaborates on tasks."""
    logger.info(f"Research team is collaborating on tasks. state: {state}")
    current_plan = state.get("current_plan")
    if not current_plan or not current_plan.steps:
        return Command(goto="planner")
    if all(step.execution_res for step in current_plan.steps):
        return Command(goto="planner")
    for step in current_plan.steps:
        if not step.execution_res:
            break
    if step.step_type and step.step_type == StepType.RESEARCH:
        return Command(goto="researcher")
    # if step.step_type and step.step_type == StepType.PROCESSING:
    #     return Command(goto="coder")
    return Command(goto="planner")


async def _execute_agent_step(
        state: State, agent, agent_name: str
) -> Command[Literal["research_team"]]:
    """Helper function to execute a step using the specified agent."""
    current_plan = state.get("current_plan")
    observations = state.get("observations", [])

    # Find the first unexecuted step
    for step in current_plan.steps:
        if not step.execution_res:
            break

    logger.info(f"Executing step: {step.title}")

    # Prepare the input for the agent
    agent_input = {
        "messages": [
            HumanMessage(
                content=f"#Task\n\n##title\n\n{step.title}\n\n##description\n\n{step.description}\n\n##locale\n\n{state.get('locale', 'zh-CN')}"
            )
        ]
    }

    # Add citation reminder for researcher agent
    if agent_name == "researcher":
        agent_input["messages"].append(
            HumanMessage(
                content="IMPORTANT: DO NOT include inline citations in the text. Instead, track all sources and include a References section at the end using link reference format. Include an empty line between each citation for better readability. Use this format for each reference:\n- [Source Title](URL)\n\n- [Another Source](URL)",
                name="system",
            )
        )

    # Invoke the agent
    result = await agent.ainvoke(input=agent_input)

    # Process the result
    response_content = result["messages"][-1].content
    logger.debug(f"{agent_name.capitalize()} full response: {response_content}")

    # Update the step with the execution result
    step.execution_res = response_content
    logger.info(f"Step '{step.title}' execution completed by {agent_name}")

    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response_content,
                    name=agent_name,
                )
            ],
            "observations": observations + [response_content],
        },
        goto="research_team",
    )


async def _setup_and_execute_agent_step(
        state: State,
        config: RunnableConfig,
        agent_type: str,
        default_agent,
        default_tools: list,
) -> Command[Literal["research_team"]]:
    """Helper function to set up an agent with appropriate tools and execute a step.

    This function handles the common logic for both researcher_node and coder_node:
    1. Configures MCP servers and tools based on agent type
    2. Creates an agent with the appropriate tools or uses the default agent
    3. Executes the agent on the current step

    Args:
        state: The current state
        config: The runnable config
        agent_type: The type of agent ("researcher" or "coder")
        default_agent: The default agent to use if no MCP servers are configured
        default_tools: The default tools to add to the agent

    Returns:
        Command to update state and go to research_team
    """
    configurable = Configuration.from_runnable_config(config)
    mcp_servers = {}
    enabled_tools = {}

    # Extract MCP server configuration for this agent type
    if configurable.mcp_settings:
        for server_name, server_config in configurable.mcp_settings["servers"].items():
            if (
                    server_config["enabled_tools"]
                    and agent_type in server_config["add_to_agents"]
            ):
                mcp_servers[server_name] = {
                    k: v
                    for k, v in server_config.items()
                    if k in ("transport", "command", "args", "url", "env")
                }
                for tool_name in server_config["enabled_tools"]:
                    enabled_tools[tool_name] = server_name
    return await _execute_agent_step(state, default_agent, agent_type)


async def researcher_node(
        state: State, config: RunnableConfig
) -> Command[Literal["research_team"]]:
    """Researcher node that do research"""
    logger.info("Researcher node is researching.")
    return await _setup_and_execute_agent_step(
        state,
        config,
        "researcher",
        research_agent,
        [web_search_tool, crawl_tool],
    )


def reporter_node(state: State):
    """Reporter node that write a final report."""
    logger.info("Reporter write final report")
    current_plan = state.get("current_plan")
    input_ = {
        "messages": [
            HumanMessage(
                f"# Research Requirements\n\n## Task\n\n{current_plan.title}\n\n## Description\n\n{current_plan.thought}"
            )
        ],
        "locale": state.get("locale", "zh-CN"),
    }
    invoke_messages = apply_prompt_template("reporter", input_)
    observations = state.get("observations", [])

    # Add a reminder about the new report format, citation style, and table usage
    invoke_messages.append(
        HumanMessage(
            content="IMPORTANT: Structure your report according to the format in the prompt. Remember to include:\n\n1. Key Points - A bulleted list of the most important findings\n2. Overview - A brief introduction to the topic\n3. Detailed Analysis - Organized into logical sections\n4. Survey Note (optional) - For more comprehensive reports\n5. Key Citations - List all references at the end\n\nFor citations, DO NOT include inline citations in the text. Instead, place all citations in the 'Key Citations' section at the end using the format: `- [Source Title](URL)`. Include an empty line between each citation for better readability.\n\nPRIORITIZE USING MARKDOWN TABLES for data presentation and comparison. Use tables whenever presenting comparative data, statistics, features, or options. Structure tables with clear headers and aligned columns. Example table format:\n\n| Feature | Description | Pros | Cons |\n|---------|-------------|------|------|\n| Feature 1 | Description 1 | Pros 1 | Cons 1 |\n| Feature 2 | Description 2 | Pros 2 | Cons 2 |",
            name="system",
        )
    )

    for observation in observations:
        invoke_messages.append(
            HumanMessage(
                content=f"Below are some observations for the research task:\n\n{observation}",
                name="observation",
            )
        )
    logger.debug(f"Current invoke messages: {invoke_messages}")
    response = get_llm().invoke(invoke_messages)
    response_content = response.content
    logger.info(f"reporter response: {response_content}")

    return {"final_report": response_content}
