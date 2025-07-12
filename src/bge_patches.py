"""
BGE-specific patches for infinity_emb compatibility
Fixes tokenization issues with BAAI/bge-code-v1 and other BGE models
"""

import logging
from typing import List, Optional, Any
import numpy as np

logger = logging.getLogger(__name__)

class BGETokenizerPatch:
    """Patch for BGE models tokenization issues in infinity_emb"""
    
    def __init__(self, original_tokenizer):
        self.original_tokenizer = original_tokenizer
        self.model_name = getattr(original_tokenizer, 'name_or_path', 'unknown')
        logger.info(f"Applying BGE tokenizer patch for: {self.model_name}")
    
    def tokenize_lengths(self, sentences: List[str], *args, **kwargs) -> List[int]:
        """
        Patched tokenize_lengths method that handles BGE models properly
        """
        try:
            # Try original method first
            result = self.original_tokenizer.tokenize_lengths(sentences, *args, **kwargs)
            if result is not None:
                return result
        except Exception as e:
            logger.warning(f"Original tokenize_lengths failed: {e}")
        
        # Fallback: manual tokenization
        try:
            logger.debug("Using fallback tokenization for BGE model")
            lengths = []
            
            for sentence in sentences:
                if not sentence or not isinstance(sentence, str):
                    lengths.append(0)
                    continue
                
                # Clean sentence for BGE models
                clean_sentence = self._clean_sentence(sentence)
                
                # Use HuggingFace tokenizer directly
                if hasattr(self.original_tokenizer, 'tokenizer'):
                    # If it's wrapped tokenizer
                    tokenizer = self.original_tokenizer.tokenizer
                elif hasattr(self.original_tokenizer, '_tokenizer'):
                    tokenizer = self.original_tokenizer._tokenizer
                else:
                    # Direct tokenizer
                    tokenizer = self.original_tokenizer
                
                # Safe tokenization
                try:
                    encoded = tokenizer.encode(clean_sentence, add_special_tokens=True)
                    lengths.append(len(encoded))
                except Exception as e:
                    logger.warning(f"Tokenization failed for sentence: {e}")
                    # Estimate length (rough approximation)
                    estimated_length = max(1, len(clean_sentence.split()) + 2)  # +2 for special tokens
                    lengths.append(estimated_length)
            
            logger.debug(f"Fallback tokenization successful: {lengths}")
            return lengths
            
        except Exception as e:
            logger.error(f"Fallback tokenization failed: {e}")
            # Last resort: return estimated lengths
            return [max(1, len(str(s).split()) + 2) for s in sentences]
    
    def _clean_sentence(self, sentence: str) -> str:
        """Clean sentence for BGE model compatibility"""
        if not sentence:
            return "default"
        
        # Remove problematic characters that might cause tokenization issues
        try:
            # Ensure UTF-8 compatibility
            clean = sentence.encode('utf-8', errors='ignore').decode('utf-8')
            
            # Remove control characters except common whitespace
            clean = ''.join(char for char in clean if ord(char) >= 32 or char in '\t\n\r')
            
            # Ensure not empty
            if not clean.strip():
                clean = "default"
            
            return clean.strip()
        except Exception:
            return "default"
    
    def __getattr__(self, name):
        """Delegate other methods to original tokenizer"""
        return getattr(self.original_tokenizer, name)


class BGESentenceTransformerPatch:
    """Patch for SentenceTransformer wrapper in infinity_emb"""
    
    def __init__(self, original_transformer):
        self.original_transformer = original_transformer
        self.model_name = getattr(original_transformer, 'model_name', 'unknown')
        
        # Apply tokenizer patch
        if hasattr(original_transformer, '_tokenizer'):
            original_transformer._tokenizer = BGETokenizerPatch(original_transformer._tokenizer)
        elif hasattr(original_transformer, 'tokenizer'):
            original_transformer.tokenizer = BGETokenizerPatch(original_transformer.tokenizer)
        
        logger.info(f"Applied BGE SentenceTransformer patch for: {self.model_name}")
    
    def tokenize_lengths(self, sentences: List[str], *args, **kwargs) -> List[int]:
        """Patched tokenize_lengths with BGE-specific handling"""
        try:
            if hasattr(self.original_transformer, '_tokenizer'):
                return self.original_transformer._tokenizer.tokenize_lengths(sentences, *args, **kwargs)
            elif hasattr(self.original_transformer, 'tokenizer'):
                return self.original_transformer.tokenizer.tokenize_lengths(sentences, *args, **kwargs)
            else:
                # Fallback to direct implementation
                return BGETokenizerPatch(self.original_transformer).tokenize_lengths(sentences, *args, **kwargs)
        except Exception as e:
            logger.error(f"BGE tokenize_lengths patch failed: {e}")
            # Emergency fallback
            return [max(1, len(str(s).split()) + 2) for s in sentences]
    
    def __getattr__(self, name):
        """Delegate other methods to original transformer"""
        return getattr(self.original_transformer, name)


def apply_bge_patches(model_instance, model_name: str):
    """
    Apply BGE-specific patches to infinity_emb model instance
    """
    if 'bge' not in model_name.lower():
        logger.debug(f"Skipping BGE patches for non-BGE model: {model_name}")
        return model_instance
    
    logger.info(f"Applying BGE patches for model: {model_name}")
    
    try:
        # Patch at multiple levels to ensure compatibility
        if hasattr(model_instance, '_model'):
            # Patch the underlying model
            model_instance._model = BGESentenceTransformerPatch(model_instance._model)
        
        if hasattr(model_instance, 'model'):
            # Patch direct model reference  
            model_instance.model = BGESentenceTransformerPatch(model_instance.model)
        
        # Patch tokenize_lengths method directly if it exists
        if hasattr(model_instance, 'tokenize_lengths'):
            original_method = model_instance.tokenize_lengths
            
            def patched_tokenize_lengths(*args, **kwargs):
                try:
                    result = original_method(*args, **kwargs)
                    if result is None:
                        raise ValueError("Original method returned None")
                    return result
                except Exception as e:
                    logger.warning(f"Using BGE fallback tokenization: {e}")
                    sentences = args[0] if args else []
                    return BGETokenizerPatch(None).tokenize_lengths(sentences)
            
            model_instance.tokenize_lengths = patched_tokenize_lengths
        
        logger.info("BGE patches applied successfully")
        return model_instance
        
    except Exception as e:
        logger.error(f"Failed to apply BGE patches: {e}")
        return model_instance


# BGE model configuration overrides
BGE_MODEL_CONFIGS = {
    "BAAI/bge-code-v1": {
        "batch_size": 32,  # Use smaller batch size for problematic models
        # Remove unsupported parameters for EngineArgs
    },
    "BAAI/bge-small-en-v1.5": {
        "batch_size": 32,
    },
    "BAAI/bge-large-en-v1.5": {
        "batch_size": 16, 
    }
}

def get_bge_model_config(model_name: str) -> dict:
    """Get BGE-specific model configuration"""
    config = BGE_MODEL_CONFIGS.get(model_name, {})
    logger.debug(f"BGE config for {model_name}: {config}")
    return config 