import httpx
import logging
import yaml
from prometheus_api_client import PrometheusConnect
from pathlib import Path
from pkg.copilot.DP_logic.extractor import extract_promql_from_response
from pkg.copilot.DP_logic.syntax_validator import validate_promql


# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Load Ollama config
def load_ollama_config(path="config/ollama_config.yaml"):
    if not Path(path).exists():
        raise FileNotFoundError(f"Ollama config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)
    
def enhanced_prompt_builder(user_prompt):
    from pkg.copilot.DP_logic.DynamicPrompt.dynamic_prompt.prompt_builder import PromptBuilder
    from pkg.copilot.DP_logic.DynamicPrompt.dynamic_prompt.retriever import Retriever

    question = user_prompt.strip()
    context = Retriever().query(question)

    prompt = PromptBuilder() \
        .with_context(context) \
        .with_user_question(question) \
        .with_overrides() \
        .with_golden_examples() \
        .with_additional_info() \
        .build()

    return prompt

OLLAMA_CONFIG = load_ollama_config()
OLLAMA_URL = OLLAMA_CONFIG.get("ollama_url", "http://localhost:11434/api/generate")
OLLAMA_MODEL = OLLAMA_CONFIG.get("ollama_model", "mistral")


# STEP 1: Ask Ollama to convert NL → PromQL
def get_promql_from_ollama(question: str) -> tuple:
    enhanced_prompt = enhanced_prompt_builder(question)
    
    logger.info(f"Sending Query to Ollama: {enhanced_prompt}")

    try:
        response = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": enhanced_prompt,
            "stream": False
        }, timeout=120)
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
        raise

    if response.status_code != 200:
        logger.error(f"Ollama failed: {response.status_code} - {response.text}")
        raise RuntimeError(f"Ollama error: {response.status_code}")

    full_response = response.json().get("response", "")
    logger.info(f"Ollama response: {full_response}\n\n")

    # Extract PromQL using the multi-pattern extractor (handles cleanup too)
    promql = extract_promql_from_response(full_response)

    # Run syntax checks — logs warnings but does not raise (let Prometheus surface hard errors)
    validate_promql(promql)

    logger.info(f"Extracted PromQL: {promql}")
    return promql, full_response

# STEP 2: Run PromQL on Prometheus
def query_prometheus(promql: str, prom_config: dict):
    logger.info(f"Querying Prometheus with: {promql}")
    # Handle both direct config and prometheus_instances structure
    if "prometheus_instances" in prom_config:
        # Extract first instance from list
        instance = prom_config["prometheus_instances"][0]
        base_url = instance["base_url"]
        disable_ssl = instance.get("disable_ssl", False)
    else:
        # Fallback logic: check for 'base_url', then 'prometheus_url', then 'url'
        base_url = prom_config.get("base_url") or \
                   prom_config.get("prometheus_url") or \
                   prom_config.get("url")
        disable_ssl = prom_config.get("disable_ssl", True)
    
    prom = PrometheusConnect(
        url=base_url,
        disable_ssl=disable_ssl
    )

    try:
        result = prom.custom_query(query=promql)
        logger.info("Prometheus query successful")
        return {
            "promql": promql,
            "result": result
        }
    except Exception as e:
        logger.error(f"Prometheus query failed: {e}")
        return {
            "promql": promql,
            "error": str(e)
        }

# STEP 3: Send PromQL results back to Ollama for final answer
def get_final_answer_from_ollama(user_question: str, promql: str, prom_result: dict) -> str:
    system_prompt = """You are an expert copilot for Prometheus metric data. Your task is to analyze Prometheus query results and provide a clear, concise answer to the user's question.

The user asked a question, we generated a PromQL query to get the data, and now you need to interpret the results to answer their original question.

Focus on:
- Directly answering the user's question based on the data
- Extracting relevant insights from the Prometheus results for the user's question
- Explaining what the metrics show in plain language
- Being concise

If there's an error or no data, explain what that means in context of their question."""

    # Format the prompt with the user question, PromQL, and results
    if "error" in prom_result:
        data_section = f"Error occurred while querying Prometheus: {prom_result['error']}"
    else:
        data_section = f"Prometheus returned the following data: {prom_result['result']}"
    
    final_prompt = f"""{system_prompt}

User's original question: {user_question}

PromQL query used: {promql}

{data_section}

Please provide a clear answer to the user's question based on this information:"""

    logger.info(f"Sending final analysis request to Ollama. User question: '{user_question}'. Prompt (truncated): '{final_prompt[:100]}...'")

    try:
        response = httpx.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": final_prompt,
            "stream": False
        }, timeout=120)
    except Exception as e:
        logger.error(f"Failed to connect to Ollama for final answer: {e}")
        raise

    if response.status_code != 200:
        logger.error(f"Ollama failed for final answer: {response.status_code} - {response.text}")
        raise RuntimeError(f"Ollama error: {response.status_code}")

    final_answer = response.json().get("response", "")
    logger.info("Final answer generated successfully")
    return final_answer

# MAIN ENTRY POINT
def run(question: str, prom_config: dict):
    try:
        promql, ollama_response = get_promql_from_ollama(question)
        result = query_prometheus(promql, prom_config)
        final_answer = get_final_answer_from_ollama(question, promql, result)
        
        result["ollama_response"] = ollama_response
        result["final_answer"] = final_answer
        return result
    except Exception as e:
        logger.exception("Error in copilot HTTP logic")
        return {"error": str(e)}
