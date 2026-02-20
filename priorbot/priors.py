from abc import abstractmethod
import numpy as np
from typing import Any
from enum import Enum
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
                    sample[key] = np.random.choice(value["enum"]).item()
                else:
                    raise ValueError(f"Unsupported type {value.type} for key {key}")

            samples.append(sample)

        return samples

    def sample_conditional(self, schema: dict[str, Any], observed: dict[str, Any], verbose: bool = False) -> dict[str, Any]:
        return self.sample(1, schema, verbose)[0]

class LLMPrior(Prior):
    def __init__(self, llm: LLM, manual_reasoning: bool = False):
        self.llm = llm
        self.manual_reasoning = manual_reasoning

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
        input_str = f"Given the observed features with these values: {observed}, generate a data point that conforms to the following schema: {schema}"
        sample = self._sample_impl(input_str, schema, verbose)
        return sample
    
    def _sample_impl(self, input_str: str, schema: dict[str, Any], verbose: bool) -> dict[str, Any]:
        if self.manual_reasoning:
            schema = {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "A description of the data point in plain text, along with the reasons for choosing specific values. To be completed before the rest of the object is generated."
                    },
                    **schema["properties"]
                },
                "required": ["reasoning"] + schema["required"]
            }

        output = self.llm.generate(input_str, output_type=schema, verbose=verbose)

        if self.manual_reasoning and type(output) is dict:
            output.pop("reasoning", None)
        
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
            print(itr_observed)
            key_to_discard = np.random.choice(list(itr_observed.keys()))
            itr_observed.pop(key_to_discard)
            
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

class MCMCAcceptanceFn(Enum):
    MCMCP = "MCMCP"
    BettingGame = "BettingGame"

class MCMCLLMPrior(Prior):
    """
    Use the Markov Chain Monta Carlo with People approach to sampling from the LLM. This uses the LLM to decide whether
    candidates in an MCMC chain should be accepted or rejected. This method relies on the proposal distribution being
    approximately uniform.
    """

    def __init__(self, llm: LLM, burn_in: int = 10, thinning: int = 1, manual_reasoning: bool = False, acceptance_fn: MCMCAcceptanceFn = MCMCAcceptanceFn.MCMCP):
        self.llm = llm
        self.proposal_dist = UniformPrior()
        self.burn_in = burn_in
        self.thinning = thinning
        self.manual_reasoning = manual_reasoning
        self.acceptance_fn = acceptance_fn

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

                samples.append(samples[-1])

            if verbose:
                print(f"Generated {len(samples[self.burn_in::self.thinning][:n_samples])}/{n_samples} samples.")

        thinned_samples = samples[self.burn_in::self.thinning][:n_samples]
        return thinned_samples

    def _acceptance(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
        if self.acceptance_fn == MCMCAcceptanceFn.MCMCP:
            return self._acceptance_mcmcp(option1, option2, schema, observed, verbose)
        elif self.acceptance_fn == MCMCAcceptanceFn.BettingGame:
            return self._acceptance_betting_game(option1, option2, schema, observed, verbose)
        else:
            raise ValueError(f"Unsupported acceptance function {self.acceptance_fn}")

    def _acceptance_mcmcp(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
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
                "description": "A description of the reasoning behind the choice, to be completed before the choice is made."
            }

        if observed:
            input_str = format(f"Given the observed features with these values: {observed}, and the following schema: {schema}, which of the following two options is more likely to be a valid data point? Option 1: {option1}. Option 2: {option2}. Respond in the format specified by this schema: {binary_schema}.")
        else:
            input_str = format(f"Which of the following two options is more likely? Option 1: {option1}. Option 2: {option2}. Respond in the format specified by this schema: {binary_schema}.")

        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("choice") == "Option 1"

    def _acceptance_betting_game(self, option1: dict[str, Any], option2: dict[str, Any], schema: dict[str, Any], observed: dict[str, Any] | None = None, verbose: bool = False) -> bool:
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
                "description": "A description of the reasoning behind the bet, to be completed before the bet is made."
            }

        bet_value = np.round(np.random.rand() * 100, 2)

        if observed:
            input_str = format(
                f"You will be presented with two sets of feature values for a data point, along with some observed "
                f"features with these values: {observed}, and the following schema: {schema}. You have the opportunity"
                f" to place a bet of ${bet_value} that Option 1 is more plausible, which will pay out $100 if you are "
                f"correct. Your aim is to maximise profit. Option 1 is {option1} and Option 2 is {option2}. Respond in"
                f" the format specified by this schema: {binary_schema}.")
        else:
            input_str = format(
                f"You will be presented with two sets of feature values for a data point. You have the opportunity to"
                f"place a bet of ${bet_value} that Option 1 is more plausible, which will pay out $100 if you are"
                f"correct. Your aim is to maximise profit. Option 1 is {option1} and Option 2 is {option2}. Respond in"
                f" the format specified by this schema: {binary_schema}.")

        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("bet") == "Place Bet"

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

