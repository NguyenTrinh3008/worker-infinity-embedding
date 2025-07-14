import os
from dotenv import load_dotenv
from functools import cached_property

DEFAULT_BATCH_SIZE = 32
DEFAULT_BACKEND = "torch"
DEFAULT_DEVICE = "auto"  # auto-detect GPU, fallback to CPU

if not os.environ.get("INFINITY_QUEUE_SIZE"):
    # how many items can be in the queue
    os.environ["INFINITY_QUEUE_SIZE"] = "48000"


class EmbeddingServiceConfig:
    def __init__(self):
        load_dotenv()

    def _get_no_required_multi(self, name, default=None):
        out = os.getenv(name, f"{default};" * len(self.model_names)).split(";")
        out = [o for o in out if o]
        if len(out) != len(self.model_names):
            raise ValueError(
                f"Env var: {name} must have the same number of elements as MODEL_NAMES"
            )
        return out

    @cached_property
    def backend(self):
        return os.environ.get("BACKEND", DEFAULT_BACKEND)

    @cached_property
    def device(self):
        """
        Device configuration for inference.
        Supports: 'auto', 'cuda', 'cpu', or specific device like 'cuda:0'
        
        'auto' will:
        - Use CUDA if available and not explicitly disabled
        - Fall back to CPU if CUDA is not available
        """
        device = os.environ.get("DEVICE", DEFAULT_DEVICE)
        
        if device == "auto":
            # Auto-detect GPU availability
            try:
                import torch
                if torch.cuda.is_available():
                    device = "cuda"
                    print(f"GPU detected: {torch.cuda.get_device_name(0)} - Using CUDA")
                else:
                    device = "cpu"
                    print("No GPU detected - Using CPU")
            except ImportError:
                device = "cpu"
                print("PyTorch not available - Using CPU")
        
        return device

    @cached_property
    def model_names(self) -> list[str]:
        model_names = os.environ.get("MODEL_NAMES")
        if not model_names:
            raise ValueError(
                "Missing required environment variable 'MODEL_NAMES'.\n"
                "Please provide at least one HuggingFace model ID, or multiple IDs separated by a semicolon.\n"
                "Examples:\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5\n"
                "  MODEL_NAMES=BAAI/bge-small-en-v1.5;intfloat/e5-large-v2\n"
            )
        model_names = model_names.split(";")
        model_names = [model_name for model_name in model_names if model_name]
        return model_names

    @cached_property
    def batch_sizes(self) -> list[int]:
        batch_sizes = self._get_no_required_multi("BATCH_SIZES", DEFAULT_BATCH_SIZE)
        batch_sizes = [int(batch_size) for batch_size in batch_sizes]
        return batch_sizes

    @cached_property
    def dtypes(self) -> list[str]:
        dtypes = self._get_no_required_multi("DTYPES", "auto")
        return dtypes

    @cached_property
    def runpod_max_concurrency(self) -> int:
        return int(os.environ.get("RUNPOD_MAX_CONCURRENCY", 300))
