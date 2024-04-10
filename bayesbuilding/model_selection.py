import numpy as np
import pandas as pd

from bayesbuilding.wrapper import PymcWrapper


class LooSelector:
    def __init__(self, model_dict: dict[str, PymcWrapper]):
        self.model_dict = model_dict

    def sample(
        self,
        y: pd.Series,
        x: pd.DataFrame = None,
        draws: int = 1000,
        tune: int = 1000,
        chains=None,
        sample_kwargs=None,
        models: list["str"] = None,
    ):
        if models is None:
            models = self.model_dict.keys()
        elif np.any(
            ~np.array(
                [
                    True if mod in list(self.model_dict.keys()) else False
                    for mod in models
                ]
            )
        ):
            raise ValueError(
                "model name in models argument not found in model_dict keys"
            )
