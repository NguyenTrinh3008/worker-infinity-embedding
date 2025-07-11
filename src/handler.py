import runpod
from utils import create_error_response
from typing import Any
from embedding_service import EmbeddingService

# Model-specific instruction prompts configuration
MODEL_INSTRUCTIONS = {
    "BAAI/bge-code-v1": "<instruct>Given a question that consists of a mix of text and code snippets, retrieve relevant answers that also consist of a mix of text and code snippets, and can help answer the question.\n<query>",
    "BAAI/bge-large-en-v1.5": "",  # No instruction needed
    "BAAI/bge-small-en-v1.5": "",  # No instruction needed
    # Add other models as needed
}

def preprocess_embedding_input(model_name: str, embedding_input):
    """
    Preprocess embedding input based on model requirements
    Adds instruction prompts for models that require them
    """
    # Handle None or empty input
    if embedding_input is None:
        return None
    
    if not model_name or model_name not in MODEL_INSTRUCTIONS:
        return embedding_input
    
    instruction = MODEL_INSTRUCTIONS[model_name]
    if not instruction:  # Empty string means no instruction needed
        return embedding_input
    
    # Handle both string and list inputs
    if isinstance(embedding_input, str):
        return instruction + embedding_input
    elif isinstance(embedding_input, list):
        # Filter out None values and handle empty strings
        processed_list = []
        for text in embedding_input:
            if text is not None:
                processed_list.append(instruction + str(text))
        return processed_list if processed_list else None
    else:
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
    job_input = job["input"]
    if job_input.get("openai_route"):
        openai_route, openai_input = job_input.get("openai_route"), job_input.get(
            "openai_input"
        )

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
            if original_input is None:
                return create_error_response("Missing 'input' field in request").model_dump()
            
            processed_input = preprocess_embedding_input(model_name, original_input)
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
        # handle the request for reranking
        if job_input.get("query"):
            call_fn, kwargs = embedding_service.infinity_rerank, {
                "query": job_input.get("query"),
                "docs": job_input.get("docs"),
                "return_docs": job_input.get("return_docs"),
                "model_name": job_input.get("model"),
            }
        elif job_input.get("input"):
            model_name = job_input.get("model")
            original_input = job_input.get("input")
            
            # Validate input
            if original_input is None:
                return create_error_response("Missing 'input' field in request").model_dump()
            
            # Preprocess input with instruction prompt if needed
            processed_input = preprocess_embedding_input(model_name, original_input)
            if processed_input is None:
                return create_error_response("Invalid or empty input after preprocessing").model_dump()
            
            call_fn, kwargs = embedding_service.route_openai_get_embeddings, {
                "embedding_input": processed_input,  # Use processed input
                "model_name": model_name,
            }
        else:
            return create_error_response(f"Invalid input: {job}").model_dump()
    try:
        out = await call_fn(**kwargs)
        return out
    except Exception as e:
        return create_error_response(str(e)).model_dump()


if __name__ == "__main__":
    runpod.serverless.start(
        {
            "handler": async_generator_handler,
            "concurrency_modifier": lambda x: embedding_service.config.runpod_max_concurrency,
        }
    )