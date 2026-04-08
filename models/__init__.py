import importlib
from dataclasses import asdict
from typing import Any, Type


# Model name → subpackage mapping
MODEL_REGISTRY: dict[str, dict[str, str]] = {
    "colbert": {"submodule": "colbert", "cls": "ColBERT"},
    "constbert": {"submodule": "constbert", "cls": "ConstBERT"},
    "tokenpooling": {"submodule": "tokenpooling", "cls": "TokenPooling"}
}


def _load_model_components(model_name: str):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    submodule = MODEL_REGISTRY[model_name]["submodule"]
    base_pkg = f"models.{submodule}"

    config_mod = importlib.import_module(f"{base_pkg}.config")
    model_mod = importlib.import_module(f"{base_pkg}.model")
    tokenizer_mod = importlib.import_module(f"{base_pkg}.tokenizer")

    ConfigCls = getattr(config_mod, "Config")
    ModelCls = getattr(model_mod, MODEL_REGISTRY[model_name]["cls"])
    TokenizerCls = getattr(tokenizer_mod, "Tokenizer")

    return ConfigCls, ModelCls, TokenizerCls


def _build_config(ConfigCls: Type, **kwargs: Any):
    config = ConfigCls()
    raw = asdict(config)
    unknown = set(kwargs) - set(raw)
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    # Coerce string values (e.g. from CLI) to the expected field type
    for k, v in kwargs.items():
        if isinstance(v, str):
            expected = type(raw[k])
            if expected is bool:
                kwargs[k] = v.lower() in ("1", "true", "yes")
            elif expected is not str:
                kwargs[k] = expected(v)
    raw.update(kwargs)
    return ConfigCls(**raw)


def model_factory(model_name: str, **kwargs):
    ConfigCls, ModelCls, TokenizerCls = _load_model_components(model_name)
    config = _build_config(ConfigCls, **kwargs)
    tokenizer = TokenizerCls(config)
    model = ModelCls(config)
    return model, tokenizer, config
