from abc import abstractmethod
import numpy as np
from typing import Any
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
        input_str = f"Generate a data point that conforms to the following schema: {schema}"
        
        while len(samples) < n_samples:
            sample = self._sample_impl(input_str, schema, verbose)
            
            if sample:
                samples.append(sample)
            
            if verbose:
                print(f"Generated {len(samples)}/{n_samples} samples.")

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        input_str = "Given the observed features with these values: {observed}, generate a data point that conforms to the following schema: {schema}"
        sample = self._sample_impl(input_str, schema, verbose)
        return sample
    
    def _sample_impl(self, input_str: str, schema: dict[str, Any], verbose: bool) -> dict[str, Any]:
        output = self.llm.generate(input_str, output_type=schema)

        if type(output) is not dict:
            if verbose:
                print(f"LLM returned invalid output {output}. Returning empty dict.")
            return {}
        else:
            return output

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

class MCMCLLMPrior(Prior):
    """
    Use the Markov Chain Monta Carlo with People approach to sampling from the LLM. This uses the LLM to decide whether
    candidates in an MCMC chain should be accepted or rejected. This method relies on the proposal distribution being
    approximately uniform.
    """

    def __init__(self, llm: LLM, burn_in: int = 10, thinning: int = 1):
        self.llm = llm
        self.proposal_dist = UniformPrior()
        self.burn_in = burn_in
        self.thinning = thinning

    def sample(self, n_samples: int, schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        return self._sample_impl(n_samples, schema, {}, verbose)

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        samples = self._sample_impl(1, schema, observed, verbose)
        return samples[0]

    def _sample_impl(self, n_samples: int, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = [self.proposal_dist.sample_conditional(schema, observed, verbose)]

        for _ in range(self.burn_in + n_samples * self.thinning):
            candidate = self.proposal_dist.sample_conditional(schema, observed, verbose)

            if np.random.choice([True, False]):
                options = [samples[-1], candidate]
            else:
                options = [candidate, samples[-1]]

            binary_schema = {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "string",
                        "enum": ["Option 1", "Option 2"]
                    }
                }
            }

            input_str = format(f"Given the observed features with these values: {observed}, and the following schema: {schema}, which of the following two options is more likely to be a valid data point? Option 1: {options[0]}. Option 2: {options[1]}. Respond in the format specified by this schema: {binary_schema}.")
            output = self.llm.generate(input_str, binary_schema)

            if type(output) is not dict or "choice" not in output:
                if verbose:
                    print(f"LLM returned invalid output {output}. Rejecting candidate.")
                samples.append(samples[-1])
            elif output.get("choice") == "Option 1":
                samples.append(options[0])
            elif output.get("choice") == "Option 2":
                samples.append(options[1])
            else:
                if verbose:
                    print(f"LLM returned invalid output {output}. Rejecting candidate.")
                samples.append(samples[-1])

            if verbose:
                print(f"Generated {len(samples[self.burn_in::self.thinning][:n_samples])}/{n_samples} samples.")

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

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--gibbs", action="store_true", help="Whether to use Gibbs sampling.")
    args = parser.parse_args()

    schema = {
        "type": "object",
        "properties": {
            "color": {
                "type": "string",
                "enum": ["red", "green", "blue"]
            },
            "shape": {
                "type": "string",
                "enum": ["circle", "square", "triangle"]
            }
        }
    }

    from .llm import OpenAICompatLLM

    prior = LLMPrior(llm=OpenAICompatLLM(
        base_url=args.base_url,
        model_name=args.model_name,
        system_prompt="You are a helpful assistant for generating data points that conform to a given schema."
    ))

    if args.gibbs:
        prior = GibbsSamplingPrior(base_prior=prior)

    samples = prior.sample(5, schema, verbose=True)
    print(samples)
