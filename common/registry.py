class Registry:
    mapping = {
        "model_name_mapping": {},
        "tokenizer_name_mapping": {},
        "lr_scheduler_name_mapping": {},
        "scorer_name_mapping": {}
    }

    @classmethod
    def register_model_name(cls, name: str):

        def warp(model_cls):
            from models import BaseModel
            assert issubclass(
                model_cls, BaseModel
            ), f"Registered model '{model_cls.__name__}' must be a subclass of BaseModel"

            if name in cls.mapping["model_name_mapping"]:
                raise KeyError(f"Model name '{name}' is already registered")

            cls.mapping["model_name_mapping"][name] = model_cls
            return model_cls

        return warp

    @classmethod
    def register_tokenizer_name(cls, name: str):

        def warp(tokenizer_cls):
            from tokenizer import BaseTokenizer
            assert issubclass(
                tokenizer_cls, BaseTokenizer
            ), f"Registered tokenizer '{tokenizer_cls.__name__}' must be a subclass of BaseTokenizer"

            if name in cls.mapping["tokenizer_name_mapping"]:
                raise KeyError(f"Tokenizer name '{name}' is already registered")

            cls.mapping["tokenizer_name_mapping"][name] = tokenizer_cls
            return tokenizer_cls

        return warp

    @classmethod
    def register_lr_scheduler(cls, name: str):
        def wrap(lr_scheduler_func):
            if name in cls.mapping["lr_scheduler_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["lr_scheduler_name_mapping"][name]
                    )
                )
            cls.mapping["lr_scheduler_name_mapping"][name] = lr_scheduler_func
            return lr_scheduler_func

        return wrap

    @classmethod
    def register_scorer(cls, name: str):
        def wrap(score_func):
            if name in cls.mapping["scorer_name_mapping"]:
                raise KeyError(
                    "Name '{}' already registered for {}.".format(
                        name, cls.mapping["scorer_name_mapping"][name]
                    )
                )
            cls.mapping["scorer_name_mapping"][name] = score_func
            return score_func

        return wrap

    @classmethod
    def get_model_cls(cls, name: str):
        return cls.mapping["model_name_mapping"].get(name, None)

    @classmethod
    def get_tokenizer_cls(cls, name: str):
        return cls.mapping["tokenizer_name_mapping"].get(name, None)

    @classmethod
    def get_lr_scheduler_func(cls, name: str):
        return cls.mapping["lr_scheduler_name_mapping"].get(name, None)

    @classmethod
    def get_scorer(cls, name: str):
        return cls.mapping["scorer_name_mapping"].get(name, None)


registry = Registry()