from pathlib import Path
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from src.config import load_yaml_config


def _create_llm_use_conf(conf: Dict[str, Any]) -> ChatOpenAI:
    llm_type = "basic"
    llm_type_map = {
        "reasoning": conf.get("REASONING_MODEL"),
        "basic": conf.get("BASIC_MODEL"),
        "vision": conf.get("VISION_MODEL"),
    }
    llm_conf = llm_type_map.get(llm_type)
    if not llm_conf:
        raise ValueError(f"Unknown LLM type: {llm_type}")
    if not isinstance(llm_conf, dict):
        raise ValueError(f"Invalid LLM Conf: {llm_type}")
    return ChatOpenAI(**llm_conf)


def get_llm() -> ChatOpenAI:
    """
    Get LLM instance by type. Returns cached instance if available.
    """
    conf = load_yaml_config(
        str((Path(__file__).parent.parent.parent / "conf.yaml").resolve())
    )
    llm = _create_llm_use_conf(conf)
    return llm
