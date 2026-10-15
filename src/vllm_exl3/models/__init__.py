"""Model-specific EXL3 integrations.

Each module here targets one model family's checkpoint layout. The
modules register themselves with vLLM on import (loaders via
``register_model_loader``); the package root imports them from
``vllm_exl3.register()``.
"""
