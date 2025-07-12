import runpod
from utils import create_error_response
from typing import Any
from embedding_service import EmbeddingService
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Model-specific instruction prompts configuration
MODEL_INSTRUCTIONS = {
    "BAAI/bge-code-v1": "",  # Keep empty for now due to tokenization issues with infinity_emb
    "BAAI/bge-large-en-v1.5": "",  # No instruction needed
    "BAAI/bge-small-en-v1.5": "",  # No instruction needed
    # Add other models as needed
}

# Special handling for models that may have tokenization issues
PROBLEMATIC_MODELS = ["BAAI/bge-code-v1"]

def preprocess_embedding_input(model_name: str, embedding_input):
    """
    Preprocess embedding input based on model requirements
    Adds instruction prompts for models that require them
    """
    logger.debug(f"preprocess_embedding_input called with model_name={model_name}, embedding_input={embedding_input}")
    
    # Handle None or empty input
    if embedding_input is None:
        logger.debug("embedding_input is None, returning None")
        return None
    
    # Special handling for problematic models
    if model_name in PROBLEMATIC_MODELS:
        logger.debug(f"Using special handling for problematic model: {model_name}")
        # Ensure clean string input for problematic models
        if isinstance(embedding_input, list):
            # Filter and clean the input
            cleaned_input = []
            for text in embedding_input:
                if text is not None and isinstance(text, str):
                    # Remove any problematic characters
                    clean_text = ''.join(char for char in str(text) if ord(char) < 128)  # ASCII only
                    if clean_text.strip():  # Only add non-empty strings
                        cleaned_input.append(clean_text)
            return cleaned_input if cleaned_input else ["default text"]
        elif isinstance(embedding_input, str):
            # Clean the string
            clean_text = ''.join(char for char in str(embedding_input) if ord(char) < 128)
            return clean_text if clean_text.strip() else "default text"
    
    if not model_name or model_name not in MODEL_INSTRUCTIONS:
        logger.debug(f"Model {model_name} not in MODEL_INSTRUCTIONS, returning original input")
        return embedding_input
    
    instruction = MODEL_INSTRUCTIONS[model_name]
    if not instruction:  # Empty string means no instruction needed
        logger.debug(f"No instruction needed for {model_name}")
        return embedding_input
    
    # Handle both string and list inputs
    if isinstance(embedding_input, str):
        result = instruction + embedding_input
        logger.debug(f"String input processed: {result[:100]}...")
        return result
    elif isinstance(embedding_input, list):
        # Filter out None values and handle empty strings
        processed_list = []
        for text in embedding_input:
            if text is not None:
                processed_list.append(instruction + str(text))
        result = processed_list if processed_list else None
        logger.debug(f"List input processed: {len(processed_list) if processed_list else 0} items")
        return result
    else:
        logger.debug(f"Unknown input type: {type(embedding_input)}")
        return embedding_input

# Gracefully catch configuration errors (e.g. missing env vars) so the user sees
# a clean message instead of a full Python traceback when the container starts.
try:
    embedding_service = EmbeddingService()
except Exception as e:  # noqa: BLE001  (intercept everything on startup)
    import sys

    sys.stderr.write(f"\nstartup failed: {e}\n")
    sys.exit(1)


async def async_generator_handler(job: dict[str, Any]):
    """Handle the requests and embedding/rerank them asynchronously."""
    logger.debug(f"async_generator_handler called with job: {job}")
    
    job_input = job["input"]
    logger.debug(f"job_input: {job_input}")
    
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get(
            "openai_input"
        )
        logger.debug(f"OpenAI route: {openai_route}, openai_input: {openai_input}")

        if openai_route and openai_route == "/v1/models":
            call_fn, kwargs = embedding_service.route_openai_models, {}
        elif openai_route and openai_route == "/v1/embeddings":
            model_name = openai_input.get("model")
            if not openai_input:
                return create_error_response("Missing input").model_dump()
            if not model_name:
                return create_error_response(
                    "Did not specify model in openai_input"
                ).model_dump()
            
            # Preprocess input with instruction prompt if needed
            original_input = openai_input.get("input")
            logger.debug(f"Original input from openai_input: {original_input}")
            
            if original_input is None:
                return create_error_response("Missing 'input' field in request").model_dump()
            
            processed_input = preprocess_embedding_input(model_name, original_input)
            logger.debug(f"Processed input: {processed_input}")
            
            if processed_input is None:
                return create_error_response("Invalid or empty input after preprocessing").model_dump()
            
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": processed_input,  # Use processed input
                "model_name": model_name,
                "return_as_list": True,
            }
        else:
            return create_error_response(
                f"Invalid OpenAI Route: {openai_route}"
            ).model_dump()
    else:
        logger.debug("Standard route (non-OpenAI)")
        # handle the request for reranking
        if job_input.get("query"):
            logger.debug("Rerank request detected")
            call_fn, kwargs = embedding_service.infinity_rerank, {
                "query": job_input.get("query"),
                "docs": job_input.get("docs"),
                "return_docs": job_input.get("return_docs"),
                "model_name": job_input.get("model"),
            }
        elif job_input.get("input"):
            logger.debug("Embedding request detected")
            model_name = job_input.get("model")
            original_input = job_input.get("input")
            
            logger.debug(f"Model: {model_name}, Original input: {original_input}")
            
            # Validate input
            if original_input is None:
                return create_error_response("Missing 'input' field in request").model_dump()
            
            # Preprocess input with instruction prompt if needed
            processed_input = preprocess_embedding_input(model_name, original_input)
            logger.debug(f"Processed input: {processed_input}")
            
            if processed_input is None:
                return create_error_response("Invalid or empty input after preprocessing").model_dump()
            
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": processed_input,  # Use processed input
                "model_name": model_name,
            }
        else:
            return create_error_response(f"Invalid input: {job}").model_dump()
    
    logger.debug(f"About to call {call_fn.__name__} with kwargs: {kwargs}")
    
    try:
        out = await call_fn(**kwargs)
        logger.debug(f"Function call successful, output type: {type(out)}")
        return out
    except Exception as e:
        logger.error(f"Exception in {call_fn.__name__}: {str(e)}", exc_info=True)
        return create_error_response(str(e)).model_dump()


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            "concurrency_modifier": lambda x: embedding_service.config.runpod_max_concurrency,
        }
    )