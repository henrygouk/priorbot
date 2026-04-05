from abc import abstractmethod
from collections.abc import Callable
import json
import numpy as np
from typing import Any, cast
from tqdm import tqdm
from .llm import LLM

class Prior:
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        pass

class UniformPrior(Prior):
    def __init__(self):
        pass

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        samples = []

        for _ in tqdm(range(n_samples), disable=not pbar, dynamic_ncols=True):
            sample = {}

            for key, value in schema["properties"].items():
                val_type = value["type"]
                if val_type == "string" and "enum" in value:
                    sample[key] = np.random.choice(value["enum"]).item()
                elif val_type == "integer":
                    sample[key] = np.random.randint(value.get("minimum", 0), value.get("maximum", 100))
                elif val_type == "number":
                    sample[key] = np.random.uniform(value.get("minimum", 0), value.get("maximum", 100))
                else:
                    raise ValueError(f"Unsupported type {val_type} for key {key}")

            samples.append(sample)

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.sample(1, schema, verbose)[0]

class GaussianPrior(Prior):
    def __init__(self, mean: float = 0.0, std: float = 1.0):
        self.mean = mean
        self.std = std

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        samples = []
        for _ in tqdm(range(n_samples), disable=not pbar, dynamic_ncols=True):
            sample = {}
            for key, value in schema["properties"].items():
                val_type = value["type"]
                if val_type == "number" or val_type == "integer":
                    sample[key] = np.random.normal(self.mean, self.std)
                else:
                    raise ValueError(f"Unsupported type {val_type} for key {key}")
            samples.append(sample)
        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.sample(1, schema, verbose)[0]

class LLMPrior(Prior):
    def __init__(self, llm: LLM, manual_reasoning: bool = False):
        self.llm = llm

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        samples = []
        input_str = f"Generate a data point that conforms to the following schema: {json.dumps(schema)}"
        pbar_samples = tqdm(total=n_samples, disable=not pbar, dynamic_ncols=True)
        while len(samples) < n_samples:
            sample = self._sample_impl(input_str, schema, verbose)
            
            if sample:
                samples.append(sample)
                pbar_samples.update(1)

                if verbose:
                    print(f"Generated {len(samples)}/{n_samples} samples.")

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        input_str = f"Given the observed features with these values: {json.dumps(observed)}, generate a data point that conforms to the following schema: {json.dumps(schema)}"
        sample = self._sample_impl(input_str, schema, verbose)
        return sample

    def _sample_impl(self, input_str: str, schema: dict[str, Any], verbose: bool) -> dict[str, Any]:
        output = self.llm.generate(input_str, schema=schema, verbose=verbose)

        if type(output) is not dict:
            if verbose:
                print(f"LLM returned invalid output {output}. Returning empty dict.")
            return {}
        else:
            return output

class GibbsLLMPrior(Prior):
    def __init__(self, base_prior: Prior, burn_in: int = 10, thinning: int = 1):
        self.base_prior = base_prior
        self.burn_in = burn_in
        self.thinning = thinning

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        return self._sample_impl(n_samples, schema, {}, verbose, pbar)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        samples = self._sample_impl(1, schema, observed, verbose, False)
        return samples[0]

    def _sample_impl(self, n_samples: int, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        samples = self.base_prior.sample(1, schema, verbose, False)

        for _ in tqdm(range(self.burn_in + n_samples * self.thinning), disable=not pbar, dynamic_ncols=True):
            itr_observed = samples[-1].copy()
            keys = list(itr_observed.keys())
            np.random.shuffle(keys)
            itr_observed = {k: itr_observed[k] for k in keys[:-1]}
            key_to_discard = keys[-1]

            itr_schema = {
                "type": "object",
                "properties": {
                    key_to_discard: schema["properties"][key_to_discard]
                },
                "required": [key_to_discard]
            }

            all_observed = {**itr_observed, **observed}
            new_marginal = self.base_prior.sample_conditional(itr_schema, all_observed, verbose)
            new_sample = itr_observed | new_marginal
            samples.append(new_sample)

            if verbose:
                print(f"Generated {len(samples[self.burn_in::self.thinning][:n_samples])}/{n_samples} samples.")
                print(f"Current sample: {samples[-1]}")

        thinned_samples = samples[self.burn_in::self.thinning][:n_samples]
        return thinned_samples

class MCMCLLMPrior(Prior):
    """
    Use the Markov Chain Monta Carlo with People approach to sampling from the LLM. This uses the LLM to decide whether
    candidates in an MCMC chain should be accepted or rejected. This method relies on the proposal distribution being
    approximately uniform.
    """

    def __init__(self, llm: LLM, burn_in: int = 10, thinning: int = 1, manual_reasoning: bool = False):
        self.llm = llm
        self.discrete_proposal_dist = UniformPrior()
        self.continuous_proposal_dist = UniformPrior()
        self.burn_in = burn_in
        self.thinning = thinning
        self.manual_reasoning = manual_reasoning

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        return self._sample_impl(n_samples, schema, {}, verbose, pbar)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        samples = self._sample_impl(1, schema, observed, verbose, False)
        return samples[0]

    def _sample_impl(self, n_samples: int, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        discrete_schema = {
            "type": "object",
            "properties": {key: value for key, value in schema["properties"].items() if value["type"] == "string"},
            "required": [key for key, value in schema["properties"].items() if value["type"] == "string"]
        }

        continuous_schema = {
            "type": "object",
            "properties": {key: value for key, value in schema["properties"].items() if value["type"] == "number" or value["type"] == "integer"},
            "required": [key for key, value in schema["properties"].items() if value["type"] == "number" or value["type"] == "integer"]
        }

        if any(value["type"] == "string" for value in schema["properties"].values()):
            disc = self.discrete_proposal_dist.sample_conditional(discrete_schema, observed, verbose)
        else:
            disc = {}

        if any(value["type"] == "number" or value["type"] == "integer" for value in schema["properties"].values()):
            # means_prompt = f"Given the following schema: {json.dumps(schema)} provide a reasonable estimate for the population means for the " \
            #     f"continuous features. Respond in JSON in the format {json.dumps(continuous_schema)}."
            #
            # stds_prompt = f"Given the following schema: {json.dumps(schema)} provide a reaonsable estimate for the population standard deviations " \
            #     f"for the continuous features. Respond in JSON in the format {json.dumps(continuous_schema)}."
            #
            # means = self.llm.generate(means_prompt, schema=continuous_schema, verbose=verbose)
            # stds = self.llm.generate(stds_prompt, schema=continuous_schema, verbose=verbose)
            # Get reasonable upper and lower bounds for the integer and number fields and use these to instantiate uniform priors
            bounds_prompt = f"Given the following schema: {json.dumps(continuous_schema)} provide reasonable estimates for the population minimum and maximum values for the continuous features. Respond in JSON in the following format: {{'feature_name': {{'min': min_value, 'max': max_value}}}} where feature_name is the name of the continuous feature, and min_value and max_value are your estimates for the population minimum and maximum values for that feature, assuming no outliers."

            bounds = self.llm.generate(bounds_prompt, schema={
                    "type": "object",
                    "properties": {
                        key: {
                            "type": "object",
                            "properties": {
                                "min": {"type": continuous_schema["properties"][key]["type"]},
                                "max": {"type": continuous_schema["properties"][key]["type"]}
                            },
                            "required": ["min", "max"]
                        } for key in continuous_schema["properties"].keys()
                    }
                },
                verbose=verbose
            )
            bounds = cast(dict[str, Any], bounds)

            # Put these mins and maxes back into the schema
            for key, value in bounds.items():
                if key in continuous_schema["properties"]:
                    continuous_schema["properties"][key]["minimum"] = value["min"]
                    continuous_schema["properties"][key]["maximum"] = value["max"]

        cont = self.continuous_proposal_dist.sample_conditional(continuous_schema, observed, verbose)

        samples = [{**disc, **cont}]

        for _ in tqdm(range(self.burn_in + n_samples * self.thinning), disable=not pbar, dynamic_ncols=True):

            # for k in candidate_continuous.keys():
            #     candidate_continuous[k] = candidate_continuous[k] * stds[k] + samples[-1][k]
            #
            #     if continuous_schema["properties"][k].get("type") == "integer":
            #         candidate_continuous[k] = np.round(candidate_continuous[k]).item()
            #     else:
            #         candidate_continuous[k] = np.round(candidate_continuous[k], 2).item()

            candidate = {**candidate_discrete, **candidate_continuous}

            if np.random.choice([True, False]):
                options = [samples[-1], candidate]
            else:
                options = [candidate, samples[-1]]

            try:
                if verbose:
                    print(f"Current sample: {samples[-1]}, Candidate: {candidate}")

                if self._acceptance(options[0], options[1], schema, observed, verbose=verbose):
                    samples.append(options[0])
                else:
                    samples.append(options[1])
            except Exception as e:
                if verbose:
                    print(f"Error during acceptance step: {e}. Rejecting candidate.")
                raise e

                samples.append(samples[-1])

            if verbose:
                print(f"Generated {len(samples[self.burn_in::self.thinning][:n_samples])}/{n_samples} samples.")

        thinned_samples = samples[self.burn_in::self.thinning][:n_samples]
        return thinned_samples

    @abstractmethod
    def _acceptance(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
        pass

def barker_prompt_template(option1: dict[str, Any], option2: dict[str, Any], input_schema: dict[str, Any], output_schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
    if observed:
        # FIXME: the `input_schema` is not aligned with `observed`
        return f"Given the observed features with these values: {json.dumps(observed)}, and the following schema: {json.dumps(input_schema)}, which of the following two options is more likely to be a valid data point? Option 1: {json.dumps(option1)}. Option 2: {json.dumps(option2)}. Respond in the format specified by this schema: {json.dumps(output_schema)}."
    else:
        return f"Which of the following two options is more likely? Option 1: {json.dumps(option1)}. Option 2: {json.dumps(option2)}. Respond in the format specified by this schema: {json.dumps(output_schema)}."

class BarkerLLMPrior(MCMCLLMPrior):

    def __init__(
        self,
        llm: LLM,
        burn_in: int = 10,
        thinning: int = 1,
        manual_reasoning: bool = False,
        prompt_template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None], str] = barker_prompt_template,
    ):
        super().__init__(llm, burn_in, thinning, manual_reasoning)
        self.prompt_template = prompt_template

    def _acceptance(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
        binary_schema = {
            "type": "object",
            "properties": {
                "choice": {
                    "type": "string",
                    "enum": ["Option 1", "Option 2"]
                }
            }
        }

        if self.manual_reasoning:
            binary_schema["properties"]["reasoning"] = {
                "type": "string",
                "description": "A step by step explanation of the reasoning behind the decision. This should be the first field in the JSON object."
            }

        input_str = self.prompt_template(option1, option2, schema, binary_schema, observed)
        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("choice") == "Option 1"

def gambling_prompt_template(option1: dict[str, Any], option2: dict[str, Any], input_schema: dict[str, Any], output_schema: dict[str, Any], bet_value: float, observed: dict[str, Any] | None = None) -> str:
    if observed:
        return (
            "You will be presented with two sets of feature values for a data point, along with some observed "
            f"features with these values: {json.dumps(observed)}, and the following schema: {json.dumps(input_schema)}. One of these is real and the other is fake. You have the opportunity"
            f" to place a bet of ${bet_value} that Option 1 is more plausible, which will pay out $100 if you are "
            f"correct. Your aim is to maximise profit. Option 1 is {json.dumps(option1)} and Option 2 is {json.dumps(option2)}. Respond with"
            f" JSON that conforms to this schema: {json.dumps(output_schema)}."
        )
    else:
        return (
            "You will be presented with two sets of feature values for a data point. One of these is real and the other is fake. You have the opportunity to "
            f"place a bet of ${bet_value} that Option 1 is more plausible, which will pay out $100 if you are "
            f"correct. Your aim is to maximise profit. Option 1 is {json.dumps(option1)} and Option 2 is {json.dumps(option2)}. Respond with"
            f" JSON that conforms to this schema: {json.dumps(output_schema)}."
        )

class GamblingLLMPrior(MCMCLLMPrior):
    def __init__(
        self,
        llm: LLM,
        burn_in: int = 10,
        thinning: int = 1,
        manual_reasoning: bool = False,
        prompt_template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], float, dict[str, Any] | None], str] = gambling_prompt_template,
    ):
        super().__init__(llm, burn_in, thinning, manual_reasoning)
        self.prompt_template = prompt_template

    def _acceptance(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
        """
        Instead of asking the LLM to determine which point is most likely, we generate a random bet and ask the LLM which side of the bet they want to be on.
        """
        binary_schema = {
            "type": "object",
            "properties": {
                "bet": {
                    "type": "string",
                    "enum": ["Place Bet", "Do Not Place Bet"]
                }
            }
        }

        if self.manual_reasoning:
            binary_schema["properties"]["reasoning"] = {
                "type": "string",
                "description": "A step by step explanation of the reasoning behind the decision to place a bet or not. This should be the first field in the JSON object."
            }

        bet_value = np.round(np.random.rand() * 100, 2)

        input_str = self.prompt_template(option1, option2, schema, binary_schema, bet_value, observed)
        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("bet") == "Place Bet"

class SplitJointConditionalPrior(Prior):
    def __init__(self, joint_prior: Prior, conditional_prior: Prior):
        self.joint_prior = joint_prior
        self.conditional_prior = conditional_prior

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        return self.joint_prior.sample(n_samples, schema, verbose, pbar)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.conditional_prior.sample_conditional(schema, observed, verbose)

class EmpiricalPrior(Prior):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    @staticmethod
    def from_prior(base_prior: Prior, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> 'EmpiricalPrior':
        samples = base_prior.sample(n_samples, schema, verbose)
        return EmpiricalPrior(samples)

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False, pbar: bool = False) -> list[dict[str, Any]]:
        filtered_samples = []
        for _ in tqdm(range(n_samples), disable=not pbar, dynamic_ncols=True):
            sample = self.samples[np.random.randint(0, len(self.samples))]
            filtered_sample = {key: value for key, value in sample.items() if key in schema["properties"]}
            filtered_samples.append(filtered_sample)

        return filtered_samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        # Can't do conditional sampling properly---we will just have to ignore conditioned observations
        return self.sample(n_samples=1, schema=schema, verbose=verbose)[0]
