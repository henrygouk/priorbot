from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
import json
import numpy as np
from typing import Any, cast
from tqdm import tqdm
from .llm import LLM


class Prior(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        pass

    @abstractmethod
    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        pass

    @abstractmethod
    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        pass


class UniformPrior(Prior):
    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        samples_dict: dict[str, np.ndarray] = {}
        for key, value in schema["properties"].items():
            val_type = value["type"]
            if val_type == "string" and "enum" in value:
                samples_dict[key] = np.random.choice(value["enum"], size=n_samples)
            elif val_type == "integer":
                assert value.get("minimum") is not None and value.get("maximum") is not None
                samples_dict[key] = np.random.randint(value["minimum"], value["maximum"], size=n_samples)
            elif val_type == "number":
                assert value.get("minimum") is not None and value.get("maximum") is not None
                samples_dict[key] = np.random.uniform(value["minimum"], value["maximum"], size=n_samples)
            else:
                raise ValueError(f"Unsupported type {val_type} for key {key}")

        features = samples_dict.keys()
        return [{k: v.item() for k, v in zip(features, values)} for values in zip(*samples_dict.values())]

    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        samples = []
        for s in schema:
            samples.append(self.sample(n_samples_per_schema, s, verbose, pbar))
        return samples

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        return self.sample(n_samples, schema, verbose)

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return self.sample_parallel(n_samples_per_schema, schema, verbose)


class GaussianPrior(Prior):
    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        samples_dict: dict[str, np.ndarray] = {}
        for key, value in schema["properties"].items():
            val_type = value["type"]
            if val_type == "number":
                assert value.get("mean") is not None and value.get("std") is not None
                samples_dict[key] = np.random.normal(value["mean"], value["std"], size=n_samples)
            else:
                raise ValueError(f"Unsupported type {val_type} for key {key}")

        features = samples_dict.keys()
        return [{k: v.item() for k, v in zip(features, values)} for values in zip(*samples_dict.values())]

    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        samples = []
        for s in schema:
            samples.append(self.sample(n_samples_per_schema, s, verbose, pbar))
        return samples

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        return self.sample(n_samples, schema, verbose)

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return self.sample_parallel(n_samples_per_schema, schema, verbose)


class AsyncPrior(Prior, ABC):
    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        results = asyncio.run(
            self._sample_impl_async(
                n_samples_per_schema=n_samples,
                schema=[schema],
                verbose=verbose,
                pbar=pbar,
            )
        )
        return results[0]

    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return asyncio.run(
            self._sample_impl_async(
                n_samples_per_schema=n_samples_per_schema,
                schema=schema,
                verbose=verbose,
                pbar=pbar,
            )
        )

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        results = asyncio.run(
            self._sample_impl_async(
                n_samples_per_schema=n_samples,
                schema=[schema],
                observed=[observed],
                verbose=verbose,
                pbar=pbar,
            )
        )
        return results[0]

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return asyncio.run(
            self._sample_impl_async(
                n_samples_per_schema=n_samples_per_schema,
                schema=schema,
                observed=observed,
                verbose=verbose,
                pbar=pbar,
            )
        )

    async def _sample_impl_async(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]] | None = None,
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        observed = observed or [{} for _ in range(len(schema))]

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(None, self._sample_impl, n_samples_per_schema, s, o, verbose, pbar)
            for s, o in zip(schema, observed)
        ]
        results = await asyncio.gather(*tasks)
        return results

    @abstractmethod
    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        pass


class LLMPrior(AsyncPrior):
    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any]], str] | None = None,
        template_conditional: Callable[[dict[str, Any], dict[str, Any]], str] | None = None,
        manual_reasoning: bool = False,
    ):
        self.llm = llm

        def _default_llm_template(schema: dict[str, Any]) -> str:
            return f"Generate a data point that conforms to the following schema: {json.dumps(schema)}"

        def _default_llm_template_conditional(observed: dict[str, Any], schema: dict[str, Any]) -> str:
            return (
                f"Given the observed features with these values: {json.dumps(observed)}, "
                f"generate a data point that conforms to the following schema: {json.dumps(schema)}"
            )

        self.template = template or _default_llm_template
        self.template_conditional = template_conditional or _default_llm_template_conditional

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        samples = []
        for _ in range(n_samples):
            if len(observed) == 0:
                prompt = self.template(schema)
            else:
                prompt = self.template_conditional(observed, schema)

            sample = self.llm.generate(prompt, schema, verbose)

            if isinstance(sample, dict):
                samples.append(sample)
            else:
                if verbose:
                    print(f"LLM returned invalid output {sample}. Skipping.")

        return samples


class GibbsLLMPrior(AsyncPrior):
    def __init__(self, base_prior: Prior, burn_in: int, thinning: int):
        self.base_prior = base_prior
        self.burn_in = burn_in
        self.thinning = thinning

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
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
            new_marginal = self.base_prior.sample_conditional(1, itr_schema, all_observed, verbose)[0]
            new_sample = itr_observed | new_marginal
            samples.append(new_sample)

            if verbose:
                print(f"Generated {len(samples[self.burn_in::self.thinning][:n_samples])}/{n_samples} samples.")
                print(f"Current sample: {samples[-1]}")

        thinned_samples = samples[self.burn_in::self.thinning][:n_samples]
        return thinned_samples


class MCMCLLMPrior(AsyncPrior):
    """
    Use the Markov Chain Monta Carlo with People approach to sampling from the LLM. This uses the LLM to decide whether
    candidates in an MCMC chain should be accepted or rejected. This method relies on the proposal distribution being
    approximately uniform.
    """

    def __init__(
        self,
        llm: LLM,
        burn_in: int,
        thinning: int,
        manual_reasoning: bool = False,
        max_trials: int = 10,
    ):
        self.llm = llm
        self.discrete_proposal_dist = UniformPrior()
        self.continuous_proposal_dist = UniformPrior()
        self.burn_in = burn_in
        self.thinning = thinning
        self.manual_reasoning = manual_reasoning
        self.max_trials = max_trials

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
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

        has_discrete_features = any(value["type"] == "string" for value in schema["properties"].values())
        has_continuous_features = any(value["type"] == "number" or value["type"] == "integer" for value in schema["properties"].values())

        disc = {}
        if has_discrete_features:
            disc = self.discrete_proposal_dist.sample_conditional(1, discrete_schema, observed, verbose)[0]

        if has_continuous_features and any(("minimum" not in value or "maximum" not in value) for value in schema["properties"].values()):
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

        cont = {}
        if has_continuous_features:
            cont = self.continuous_proposal_dist.sample_conditional(1, continuous_schema, observed, verbose)[0]

        samples = [{**disc, **cont}]

        for _ in tqdm(range(self.burn_in + n_samples * self.thinning), disable=not pbar, dynamic_ncols=True):
            candidate = {}  # Prevent PossiblyUnboundVariable error from type checkers
            for _ in range(self.max_trials):  # Try up to max_trials times to generate a valid candidate
                candidate_discrete = {}
                if has_discrete_features:
                    candidate_discrete = self.discrete_proposal_dist.sample_conditional(
                        1, discrete_schema, observed, verbose
                    )[0]

                candidate_continuous = {}
                if has_continuous_features:
                    candidate_continuous = self.continuous_proposal_dist.sample_conditional(
                        1, continuous_schema, observed, verbose
                    )[0]

                # for k in candidate_continuous.keys():
                #     candidate_continuous[k] = candidate_continuous[k] * stds[k] + samples[-1][k]
                #
                #     if continuous_schema["properties"][k].get("type") == "integer":
                #         candidate_continuous[k] = np.round(candidate_continuous[k]).item()
                #     else:
                #         candidate_continuous[k] = np.round(candidate_continuous[k], 2).item()

                candidate = {**candidate_discrete, **candidate_continuous}

                # If the candidate is the same as the previous sample, try again
                if all(samples[-1][k] == candidate[k] for k in candidate.keys()):
                    continue
                break

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
    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
        pass


def barker_prompt_template(
    option1: dict[str, Any],
    option2: dict[str, Any],
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    observed: dict[str, Any] | None = None,
) -> str:
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

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
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


def gambling_prompt_template(
    option1: dict[str, Any],
    option2: dict[str, Any],
    input_schema: dict[str, Any],
    output_schema: dict[str, Any],
    bet_value: float,
    observed: dict[str, Any] | None = None,
) -> str:
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

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
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

        input_str = self.prompt_template(
            option1, option2, schema, binary_schema, bet_value, observed
        )
        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("bet") == "Place Bet"


class SplitJointConditionalPrior(Prior):
    def __init__(self, joint_prior: Prior, conditional_prior: Prior):
        self.joint_prior = joint_prior
        self.conditional_prior = conditional_prior

    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        return self.joint_prior.sample(n_samples, schema, verbose, pbar)

    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return self.joint_prior.sample_parallel(n_samples_per_schema, schema, verbose, pbar)

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        return self.conditional_prior.sample_conditional(n_samples, schema, observed, verbose)

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        return self.conditional_prior.sample_conditional_parallel(
            n_samples_per_schema, schema, observed, verbose
        )


class EmpiricalPrior(Prior):
    def __init__(self, samples: list[dict[str, Any]]):
        self.samples = samples

    @staticmethod
    def from_prior(
        base_prior: Prior,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
    ) -> 'EmpiricalPrior':
        samples = base_prior.sample(n_samples, schema, verbose)
        return EmpiricalPrior(samples)

    def _filter_to_schema(
        self,
        sample: dict[str, Any],
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        props = schema["properties"]
        return {k: v for k, v in sample.items() if k in props}

    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        indices = np.random.randint(0, len(self.samples), size=n_samples)
        return [self._filter_to_schema(self.samples[i], schema) for i in indices]

    def sample_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[list[dict[str, Any]]]:
        samples = []
        for s in schema:
            samples.append(self.sample(n_samples_per_schema, s, verbose, pbar))
        return samples

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        # Can't condition on observations — just draw from the marginal
        return self.sample(n_samples, schema, verbose)

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        # Can't condition on observations — just draw from the marginal
        return self.sample_parallel(n_samples_per_schema, schema, verbose)