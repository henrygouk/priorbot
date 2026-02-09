from abc import abstractmethod
import argparse
from data import Dataset, load_dataset, save_dataset
import json
import numpy as np
from typing import Any, Optional
from .llm import LLM

class Prior:
    def __init__(self):
        pass

    @abstractmethod
    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        pass

class UniformPrior(Prior):
    def __init__(self):
        pass

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = []

        for _ in range(n_samples):
            sample = {}

            for key, value in schema["properties"].items():
                if value["type"] == "string" and "enum" in value:
                    sample[key] = np.random.choice(value["enum"])
                else:
                    raise ValueError(f"Unsupported type {value.type} for key {key}")

            samples.append(sample)

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.sample(1, schema, verbose)[0]

class LLMPrior(Prior):
    def __init__(self, llm: LLM):
        self.llm = llm

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = []
        input_str = format(f"Generate a data point that conforms to the following schema: {schema}")
        
        while len(samples) < n_samples:
            sample = self._sample_impl(input_str, schema, verbose)
            
            if sample:
                samples.append(sample)
            
            if verbose:
                print(f"Generated {len(samples)}/{n_samples} samples.")

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        input_str = format(f"Given the observed features with these values: {observed}, generate a data point that conforms to the following schema: {schema}")
        sample = self._sample_impl(input_str, schema, verbose)
        return sample
    
    def _sample_impl(self, input_str: str, schema: dict[str, Any], verbose: bool) -> dict[str, Any]:
        output = self.llm.generate(input_str, schema)

        if output is None:
            if verbose:
                print("LLM returned None. Returning empty dict.")
            return {}

        try:
            result = json.loads(output)
            return result
        except json.JSONDecodeError as e:
            if verbose:
                print(f"Error decoding JSON: {e}. Output: {output}")
            return {}
        except Exception as e:
            if verbose:
                print(f"Unexpected error: {e}. Output: {output}")
            return {}

class GibbsSamplingPrior(Prior):
    def __init__(self, base_prior: Prior, burn_in: int = 10, thinning: int = 1):
        self.base_prior = base_prior
        self.burn_in = burn_in
        self.thinning = thinning

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        return self._sample_impl(n_samples, schema, {}, verbose)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        samples = self._sample_impl(1, schema, observed, verbose)
        return samples[0]

    def _sample_impl(self, n_samples: int, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = self.base_prior.sample(1, schema, verbose)

        for _ in range(self.burn_in + n_samples * self.thinning):
            itr_observed = samples[-1].copy()
            key_to_discard = np.random.choice(list(itr_observed.keys()))
            itr_observed.pop(key_to_discard)
            
            itr_schema = {
                "type": "object",
                "properties": {
                    key_to_discard: schema["properties"][key_to_discard]
                },
                "required": [key_to_discard] if key_to_discard in schema.get("required", []) else []
            }

            all_observed = {**itr_observed, **observed}
            new_sample = self.base_prior.sample_conditional(itr_schema, all_observed, verbose)
            samples.append(new_sample)

        thinned_samples = samples[self.burn_in::self.thinning][:n_samples]
        return thinned_samples

class SplitJointConditionalPrior(Prior):
    def __init__(self, joint_prior: Prior, conditional_prior: Prior):
        self.joint_prior = joint_prior
        self.conditional_prior = conditional_prior

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        return self.joint_prior.sample(n_samples, schema, verbose)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.conditional_prior.sample_conditional(schema, observed, verbose)

class EmpiricalPrior(Prior):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    @staticmethod
    def from_prior(base_prior: Prior, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> 'EmpiricalPrior':
        samples = base_prior.sample(n_samples, schema, verbose)
        return EmpiricalPrior(samples)

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = [self.samples[np.random.randint(0, len(self.samples))] for _ in range(n_samples)]

        filtered_samples = []

        for sample in samples:
            filtered_sample = {key: value for key, value in sample.items() if key in schema["properties"]}
            filtered_samples.append(filtered_sample)

        return filtered_samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        # Can't do conditional sampling properly---we will just have to ignore conditioned observations
        return self.sample(n_samples=1, schema=schema, verbose=verbose)[0]

