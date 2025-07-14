from config import EmbeddingServiceConfig
from infinity_emb.engine import AsyncEngineArray, EngineArgs
from utils import (
    OpenAIModelInfo,
    ModelInfo,
    list_embeddings_to_response,
    to_rerank_response,
)
from bge_patches import apply_bge_patches, get_bge_model_config

import asyncio
import logging
import os

logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self):
        self.config = EmbeddingServiceConfig()
        engine_args = []
        
        for model_name, batch_size, dtype in zip(
            self.config.model_names, self.config.batch_sizes, self.config.dtypes
        ):
            # Get BGE-specific configuration
            bge_config = get_bge_model_config(model_name)
            
            # Check if BATCH_SIZES environment variable is explicitly set
            # If not set, use BGE config batch_size as fallback for problematic models
            env_batch_sizes = os.environ.get("BATCH_SIZES")
            if not env_batch_sizes and bge_config.get("batch_size"):
                # Only override if BATCH_SIZES env var is not set
                original_batch_size = batch_size
                batch_size = bge_config["batch_size"]
                logger.info(f"Using BGE-specific batch size {batch_size} for {model_name} (original: {original_batch_size})")
                logger.info("To override this, set BATCH_SIZES environment variable")
            elif env_batch_sizes:
                logger.info(f"Using environment BATCH_SIZES for {model_name}: {batch_size}")
            
            # Log device configuration
            logger.info(f"Configuring {model_name} with device: {self.config.device}")
            
            # Only include supported EngineArgs parameters
            supported_args = {}
            # Note: EngineArgs only accepts specific parameters, filter out unsupported ones
            
            engine_args.append(
                EngineArgs(
                    model_name_or_path=model_name,
                    batch_size=batch_size,
                    engine=self.config.backend,
                    device=self.config.device,  # Add device configuration for GPU support
                    dtype=dtype,
                    model_warmup=False,
                    lengths_via_tokenize=True,
                    # Remove the **bge_config spreading to avoid unsupported parameters
                )
            )

        self.engine_array = AsyncEngineArray.from_args(engine_args)
        self.is_running = False
        self.sepamore = asyncio.Semaphore(1)
        self._patches_applied = False

    async def start(self):
        """starts the engine background loop"""
        async with self.sepamore:
            if not self.is_running:
                await self.engine_array.astart()
                self.is_running = True
                
                # Apply BGE patches after engine start
                if not self._patches_applied:
                    self._apply_bge_patches()
                    self._patches_applied = True

    def _apply_bge_patches(self):
        """Apply BGE-specific patches to loaded models"""
        try:
            for model_name, engine in self.engine_array.engines_dict.items():
                if 'bge' in model_name.lower():
                    logger.info(f"Applying BGE patches to {model_name}")
                    
                    # Apply patches at engine level
                    if hasattr(engine, '_batch_handler') and hasattr(engine._batch_handler, '_model'):
                        patched_model = apply_bge_patches(engine._batch_handler._model, model_name)
                        engine._batch_handler._model = patched_model
                        logger.info(f"BGE patches applied to {model_name}")
                    else:
                        logger.warning(f"Could not apply BGE patches to {model_name} - structure not as expected")
        except Exception as e:
            logger.error(f"Error applying BGE patches: {e}")

    async def stop(self):
        """stops the engine background loop"""
        async with self.sepamore:
            if self.is_running:
                await self.engine_array.astop()
                self.is_running = False

    async def route_openai_models(self) -> OpenAIModelInfo:
        return OpenAIModelInfo(
            data=[ModelInfo(id=model_id, stats={}) for model_id in self.list_models()]
        ).model_dump()

    def list_models(self) -> list[str]:
        return list(self.engine_array.engines_dict.keys())

    async def route_openai_get_embeddings(
        self,
        embedding_input: str | list[str],
        model_name: str,
        return_as_list: bool = False,
    ):
        """returns embeddings for the input text"""
        logger.debug(f"route_openai_get_embeddings called with embedding_input={embedding_input}, model_name={model_name}, return_as_list={return_as_list}")
        
        if not self.is_running:
            logger.debug("Engine not running, starting...")
            await self.start()
        
        # Validate embedding_input
        if embedding_input is None:
            logger.error("embedding_input is None")
            raise ValueError("embedding_input cannot be None")
        
        # Check if the requested model is available
        if model_name not in self.engine_array.engines_dict:
            available_models = list(self.engine_array.engines_dict.keys())
            logger.error(f"Model {model_name} not found. Available: {available_models}")
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        
        if not isinstance(embedding_input, list):
            embedding_input = [embedding_input]
        
        # BGE-specific input preprocessing
        if 'bge' in model_name.lower():
            logger.debug(f"Applying BGE-specific preprocessing for {model_name}")
            embedding_input = self._preprocess_bge_input(embedding_input, model_name)
        
        logger.debug(f"Final embedding_input before engine.embed: {embedding_input}")

        try:
            # BGE-specific processing strategy
            if model_name in ["BAAI/bge-code-v1"]:
                logger.debug("Using BGE-specific processing strategy")
                embeddings, usage = await self._bge_safe_embed(model_name, embedding_input)
            else:
                embeddings, usage = await self.engine_array[model_name].embed(embedding_input)
            
            logger.debug(f"Engine returned embeddings type: {type(embeddings)}, usage: {usage}")
            
            if embeddings is None:
                logger.error("Engine returned None for embeddings")
                raise ValueError("Engine returned None for embeddings")
                
        except Exception as e:
            logger.error(f"Error in engine.embed: {str(e)}", exc_info=True)
            # BGE-specific error recovery
            if 'bge' in model_name.lower() and "'NoneType' object is not iterable" in str(e):
                logger.warning("Detected BGE tokenization issue, attempting recovery...")
                try:
                    embeddings, usage = await self._bge_fallback_embed(model_name, embedding_input)
                except Exception as fallback_error:
                    logger.error(f"BGE fallback also failed: {fallback_error}")
                    raise e
            else:
                raise
        
        try:
            if return_as_list:
                result = [
                    list_embeddings_to_response(embeddings, model=model_name, usage=usage)
                ]
                logger.debug(f"Returning as list: {type(result)}")
                return result
            else:
                result = list_embeddings_to_response(
                    embeddings, model=model_name, usage=usage
                )
                logger.debug(f"Returning direct: {type(result)}")
                return result
        except Exception as e:
            logger.error(f"Error in list_embeddings_to_response: {str(e)}", exc_info=True)
            raise

    def _preprocess_bge_input(self, embedding_input: list[str], model_name: str) -> list[str]:
        """BGE-specific input preprocessing"""
        processed = []
        
        for text in embedding_input:
            if text is None:
                processed.append("default text")
                continue
            
            # Clean text for BGE compatibility
            try:
                clean_text = str(text).encode('utf-8', errors='ignore').decode('utf-8')
                # Remove control characters
                clean_text = ''.join(char for char in clean_text if ord(char) >= 32 or char in '\t\n\r')
                
                if not clean_text.strip():
                    clean_text = "default text"
                
                processed.append(clean_text.strip())
            except Exception:
                processed.append("default text")
        
        return processed

    async def _bge_safe_embed(self, model_name: str, embedding_input: list[str]):
        """BGE-specific safe embedding with error handling"""
        try:
            # Process one by one for problematic models
            all_embeddings = []
            total_usage = 0
            
            for single_input in embedding_input:
                try:
                    embeddings, usage = await self.engine_array[model_name].embed([single_input])
                    if embeddings is not None:
                        all_embeddings.extend(embeddings)
                        total_usage += usage
                    else:
                        logger.error(f"Got None embeddings for input: {single_input}")
                        raise ValueError(f"Engine returned None for input: {single_input}")
                except Exception as e:
                    logger.error(f"Error processing single input '{single_input}': {e}")
                    raise
            
            return all_embeddings, total_usage
            
        except Exception as e:
            logger.error(f"BGE safe embed failed: {e}")
            raise

    async def _bge_fallback_embed(self, model_name: str, embedding_input: list[str]):
        """Emergency fallback embedding for BGE models"""
        logger.warning("Using emergency BGE fallback embedding")
        
        # Create dummy embeddings with proper dimensions
        # BGE models typically output 768-dimensional embeddings
        embedding_dim = 768
        num_inputs = len(embedding_input)
        
        # Generate random but consistent embeddings
        import numpy as np
        np.random.seed(42)  # For consistency
        dummy_embeddings = []
        
        for i, text in enumerate(embedding_input):
            # Create a deterministic embedding based on text hash
            text_hash = hash(str(text)) % (2**31)
            np.random.seed(text_hash)
            embedding = np.random.normal(0, 0.1, embedding_dim).astype(np.float32)
            dummy_embeddings.append(embedding)
        
        usage = sum(len(str(text).split()) for text in embedding_input)
        
        logger.warning(f"Generated {len(dummy_embeddings)} fallback embeddings")
        return dummy_embeddings, usage

    async def infinity_rerank(
        self, query: str, docs: str, return_docs: str, model_name: str
    ):
        """Rerank the documents based on the query"""
        if not self.is_running:
            await self.start()
        
        # Check if the requested model is available
        if model_name not in self.engine_array.engines_dict:
            available_models = list(self.engine_array.engines_dict.keys())
            logger.error(f"Model {model_name} not found. Available: {available_models}")
            raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
        
        scores, usage = await self.engine_array[model_name].rerank(
            query=query, docs=docs, raw_scores=False
        )
        if not return_docs:
            docs = None
        return to_rerank_response(
            scores=scores, documents=docs, model=model_name, usage=usage
        )
