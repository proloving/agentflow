import argparse
import asyncio

from src.workflow import run_agent_workflow_async


def ask(
        question,
        debug=False,
        max_plan_iterations=1,
        max_step_num=3,
        enable_background_investigation=True,
):
    """Run the agent workflow with the given question.

    Args:
        question: The user's query or request
        debug: If True, enables debug level logging
        max_plan_iterations: Maximum number of plan iterations
        max_step_num: Maximum number of steps in a plan
        enable_background_investigation: If True, performs web search before planning to enhance context
    """
    asyncio.run(
        run_agent_workflow_async(
            user_input=question,
            debug=debug,
            max_plan_iterations=max_plan_iterations,
            max_step_num=max_step_num,
            enable_background_investigation=enable_background_investigation,
        )
    )


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the agent")
    parser.add_argument("query", nargs="*", help="The query to process")

    args = parser.parse_args()

    # Parse user input from command line arguments or user input
    if args.query:
        user_query = " ".join(args.query)
    else:
        user_query = input("输入你的问题: ")

    # Run the agent workflow with the provided parameters
    ask(
        question=user_query,
        # debug=args.debug,
        # max_plan_iterations=args.max_plan_iterations,
        # max_step_num=args.max_step_num,
        # enable_background_investigation=args.enable_background_investigation,
    )
