import pytensor.tensor as pt
from pytensor.graph import Apply, Op
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class LogLike(Op, ABC):
    def make_node(self, *variables, x, data) -> Apply:
        # Convert inputs to tensor variables
        variable_list = [pt.as_tensor(val) for val in variables]
        x = pt.as_tensor(x)
        data = pt.as_tensor(data)

        inputs = [*variable_list, x, data]
        # Define output type, in our case a vector of likelihoods
        # with the same dimensions and same data type as data
        # If data must always be a vector, we could have hard-coded
        # outputs = [pt.vector()]
        outputs = [data.type()]

        # Apply is an object that combines inputs, outputs and an Op (self)
        return Apply(self, inputs, outputs)

    def perform(
        self, node: Apply, inputs: list[np.ndarray], outputs: list[list[None]]
    ) -> None:
        # This is the method that compute numerical output
        # given numerical inputs. Everything here is numpy arrays
        variables = inputs[:-2]
        x, data = inputs[-2:]  # this will contain my variables

        # call our numpy log-likelihood function
        loglike_eval = self.loglike(*variables, x=x, data=data)

        # Save the result in the outputs list provided by PyTensor
        # There is one list per output, each containing another list
        # pre-populated with a `None` where the result should be saved.
        outputs[0][0] = np.asarray(loglike_eval)

    @abstractmethod
    def loglike(self, *variable_list, x, data):
        """
        Define the custom loglikelihood
        :param variable_list: list of numpy array containing varaibles values
        :param x: numpy array containing boundary conditions
        :param data: observation for likelyhood computation
        :return:
        """

    def custom_dist_loglike(self, data, *dist_params):
        # data, or observed is always passed as the first input of CustomDist
        variables = dist_params[:-1]
        return self.loglike(*variables, x=dist_params[-1], data=data)


class ModelicaFMULogLike(LogLike):
    def __init__(self, modelica_fmu_model):
        self.modelica_fmu_model = modelica_fmu_model
        self.variables_keys = None
        self.x_index = None
        self.c_columns = None

    def loglike(self, *variable_list, x, data):
        variable_dict = {
            key: val
            for key, val in zip(self.variables_keys, variable_list)
            if key != "sigma"
        }

        sigma = variable_list[-1]

        x = pd.DataFrame(x, columns=self.c_columns, index=self.x_index)

        if list(variable_dict.values())[0].shape == ():
            local_dict = {key: float(val) for key, val in variable_dict.items()}

            model = (
                self.modelica_fmu_model.simulate(
                    parameter_dict=local_dict,
                    # x=x
                )
                .squeeze()
                .to_numpy()
            )
        else:
            res = []
            for i in range(len(list(variable_dict.values())[0])):
                local_dict = {key: val[i] for key, val in variable_dict.items()}
                res.append(
                    self.modelica_fmu_model.simulate(
                        parameter_dict=local_dict,
                        # x=x,
                        logger=False,
                        debug_logging=False,
                    )
                )
                model = pd.concat(res).T.to_numpy()

        return (
            -0.5 * ((data - model) / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi))
            - np.log(sigma)
        )


class RefLogLike(LogLike):
    def __init__(self, model_function):
        self.model_function = model_function
        self.variables_keys = None

    def loglike(self, *variable_list, x, data):
        variable_dict = {
            key: val
            for key, val in zip(self.variables_keys, variable_list)
            if key != "sigma"
        }

        sigma = variable_list[-1]

        model = self.model_function(variable_dict, x)

        return (
            -0.5 * ((data - model) / sigma) ** 2
            - np.log(np.sqrt(2 * np.pi))
            - np.log(sigma)
        )
