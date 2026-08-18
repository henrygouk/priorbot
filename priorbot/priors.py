from abc import ABC, abstractmethod
import asyncio
from collections.abc import Callable
from copy import deepcopy
import json
import numpy as np
from typing import Any, Coroutine, cast
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

    @abstractmethod
    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        pass

    def sample_conditional_parallel(
        self,
        n_samples_per_schema: int,
        schema: list[dict[str, Any]],
        observed: list[dict[str, Any]],
        verbose: bool = False,
    ) -> list[list[dict[str, Any]]]:
        samples = []
        for s, o in zip(schema, observed):
            samples.append(self.sample_conditional(n_samples_per_schema, s, o, verbose))
        return samples


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
            elif val_type == "integer" or val_type == "number":
                if value.get("minimum") is None or value.get("maximum") is None:
                    raise ValueError(f"Minimum and maximum must be specified for numeric type {key}")
                if val_type == "integer":
                    samples_dict[key] = np.random.randint(value["minimum"], value["maximum"] + 1, size=n_samples)
                else:  # number
                    samples_dict[key] = np.random.uniform(value["minimum"], value["maximum"], size=n_samples)
            else:
                raise ValueError(f"Unsupported type {val_type} for key {key}")

        features = samples_dict.keys()
        return [{k: v.item() for k, v in zip(features, values)} for values in zip(*samples_dict.values())]

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        # Can't condition on observations — just draw from the marginal
        return self.sample(n_samples, schema, verbose)

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
                if value.get("mean") is None or value.get("std") is None:
                    raise ValueError(f"Mean and standard deviation must be specified for number type {key}")
                samples_dict[key] = np.random.normal(value["mean"], value["std"], size=n_samples)
            else:
                raise ValueError(f"Unsupported type {val_type} for key {key}")

        features = samples_dict.keys()
        return [{k: v.item() for k, v in zip(features, values)} for values in zip(*samples_dict.values())]

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        # Can't condition on observations — just draw from the marginal
        return self.sample(n_samples, schema, verbose)


class AsyncPrior(Prior, ABC):
    @staticmethod
    def _run_async(coro: Coroutine) -> Any:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        else:
            import nest_asyncio
            nest_asyncio.apply()
            return loop.run_until_complete(coro)

    def sample(
        self,
        n_samples: int,
        schema: dict[str, Any],
        verbose: bool = False,
        pbar: bool = False,
    ) -> list[dict[str, Any]]:
        results = self._run_async(
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
        return self._run_async(
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
        results = self._run_async(
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
        return self._run_async(
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
        if observed is not None and len(observed) != len(schema):
            raise ValueError(f"Number of observed samples ({len(observed)}) must match number of schemas ({len(schema)})")

        loop = asyncio.get_running_loop()
        tasks = [
            loop.run_in_executor(
                None,
                self._sample_impl,
                n_samples_per_schema,
                schema[i],
                observed[i] if observed is not None else None,
                verbose,
                i if pbar else None,
            )
            for i in range(len(schema))
        ]
        results = await asyncio.gather(*tasks)
        return results

    @abstractmethod
    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        pass


def default_llm_template(schema: dict[str, Any], observed: dict[str, Any] | None = None) -> str:
    if observed:
        return (
            f"Given the observed features with these values: {json.dumps(observed)}, "
            f"generate a data point that conforms to the following schema: {json.dumps(schema)}"
        )
    else:
        return f"Generate a data point that conforms to the following schema: {json.dumps(schema)}"


DEFAULT_REASONING_PROMPT = (
    "A step-by-step explanation of the reasoning behind the {obj}. "
    "This should be the first field in the JSON object."
)


class LLMPrior(AsyncPrior):
    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any], dict[str, Any] | None], str] = default_llm_template,
        shuffle_variables: bool = True,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="generation of the data point"),
    ):
        self.llm = llm
        self.template = template
        self.shuffle_variables = shuffle_variables
        self.manual_reasoning = manual_reasoning
        self.reasoning_prompt = reasoning_prompt

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        samples = []
        for _ in tqdm(
            range(n_samples), disable=pbar is None, position=pbar, desc=f"Worker {pbar}", dynamic_ncols=True
        ):

            if self.shuffle_variables:
                keys = list(schema["properties"].keys())
                np.random.shuffle(keys)
                schema["properties"] = {k: schema["properties"][k] for k in keys}
                schema["required"] = keys

            if self.manual_reasoning:
                gen_schema = deepcopy(schema)
                gen_schema["properties"] = {
                    "reasoning": {"type": "string", "description": self.reasoning_prompt},
                    **gen_schema["properties"],
                }
                gen_schema["required"] = ["reasoning"] + gen_schema["required"]
            else:
                gen_schema = schema

            prompt = self.template(gen_schema, observed)
            sample = self.llm.generate(prompt, gen_schema, verbose)

            if isinstance(sample, dict):
                sample.pop("reasoning", None)
                samples.append(sample)
            else:  # String should not be given as output (see .generate methods in llm.py)
                raise ValueError(f"LLM returned invalid output {sample}.")

        return samples


class GibbsLLMPrior(AsyncPrior):
    def __init__(
        self,
        llm_prior: LLMPrior,
        burn_in: int,
        thinning: int,
        block_size: int = 1,
        sweep: bool = False,
        shuffle_variables: bool = True,
    ):
        self.llm_prior = llm_prior
        # We shuffle variables in the Gibbs procedure, no need to shuffle them during sampling
        self.llm_prior.shuffle_variables = False
        self.burn_in = burn_in
        self.thinning = thinning
        self.block_size = block_size
        self.sweep = sweep
        self.shuffle_variables = shuffle_variables
        if not (self.shuffle_variables or self.sweep):
            raise ValueError("Either shuffle_variables or sweep must be True.")

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        samples = self.llm_prior.sample(1, schema, verbose, False)

        chain_length = self.burn_in + n_samples * self.thinning
        keys_pool = []
        for _ in tqdm(
            range(chain_length), disable=pbar is None, position=pbar, desc=f"Chain {pbar}", dynamic_ncols=True
        ):
            itr_observed = samples[-1].copy()
            keys = list(itr_observed.keys())
            if self.shuffle_variables:
                np.random.shuffle(keys)

            if not self.sweep:
                keys_to_discard = keys[-self.block_size:]
            else:
                if len(keys_pool) < self.block_size:
                    keys_pool = keys_pool + keys

                keys_to_discard = keys_pool[:self.block_size]
                keys_pool = keys_pool[self.block_size:]

            itr_observed = {k: itr_observed[k] for k in keys if k not in keys_to_discard}

            itr_schema = {
                "type": "object",
                "properties": {key: schema["properties"][key] for key in keys_to_discard},
                "required": keys_to_discard
            }

            all_observed = {**itr_observed, **(observed or {})}
            new_marginal = self.llm_prior.sample_conditional(1, itr_schema, all_observed, verbose)[0]
            new_sample = itr_observed | new_marginal
            samples.append(new_sample)

            if verbose:
                print(f"Generated {len(samples[self.burn_in + self.thinning::self.thinning][:n_samples])}/{n_samples} samples.")
                print(f"Current sample: {samples[-1]}")

        thinned_samples = samples[self.burn_in + self.thinning::self.thinning][:n_samples]
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
        template: Callable[..., str],
        burn_in: int,
        thinning: int,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="decision"),
    ):
        self.llm = llm
        self.template = template
        self.discrete_proposal_dist = UniformPrior()
        self.continuous_proposal_dist = UniformPrior()
        self.burn_in = burn_in
        self.thinning = thinning
        self.shuffle_variables = shuffle_variables
        self.max_trials = max_trials
        self.manual_reasoning = manual_reasoning
        self.reasoning_prompt = reasoning_prompt

    @staticmethod
    def _sample_single(
        proposal: Prior,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> dict[str, Any]:
        if observed is None:
            return proposal.sample(1, schema, verbose)[0]
        else:
            return proposal.sample_conditional(1, schema, observed, verbose)[0]

    def split_schema(self, schema: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        discrete_keys = [
            key for key, value in schema["properties"].items()
            if value["type"] == "string"
        ]
        continuous_keys = [
            key for key, value in schema["properties"].items()
            if value["type"] in ("number", "integer")
        ]
        discrete_schema = {
            "type": "object",
            "properties": {key: schema["properties"][key] for key in discrete_keys},
            "required": discrete_keys,
        }
        continuous_schema = {
            "type": "object",
            "properties": {key: schema["properties"][key] for key in continuous_keys},
            "required": continuous_keys,
        }
        return discrete_schema, continuous_schema

    def _initialize_proposal_chain(
        self,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        """Draw an initial sample for an MCMC chain whose proposals are decomposed into
        independent discrete (string) and continuous (number/integer) parts.

        The schema is split by feature type, ``llm`` is asked to estimate population
        minima/maxima for any continuous features missing them, and the supplied
        proposal distributions are then sampled (conditional on ``observed`` when
        provided). Returns ``(initial_sample, discrete_schema, continuous_schema)``;
        the returned sub-schemas have any LLM-estimated bounds populated, ready for
        reuse by later proposals in the chain. ``schema`` is not mutated.
        """
        discrete_schema, continuous_schema = self.split_schema(schema)
        needs_bounds = continuous_schema["properties"] and any(
            "minimum" not in value or "maximum" not in value
            for value in continuous_schema["properties"].values()
        )
        if needs_bounds:
            bounds_prompt = (
                f"Given the following schema: {json.dumps(continuous_schema)} provide "
                "reasonable estimates for the population minimum and maximum values for "
                "the continuous features. Respond in JSON in the following format: "
                "{'feature_name': {'min': min_value, 'max': max_value}} where "
                "feature_name is the name of the continuous feature, and min_value and "
                "max_value are your estimates for the population minimum and maximum "
                "values for that feature, assuming no outliers."
            )
            bounds_schema = {
                "type": "object",
                "properties": {
                    key: {
                        "type": "object",
                        "properties": {
                            "min": {"type": continuous_schema["properties"][key]["type"]},
                            "max": {"type": continuous_schema["properties"][key]["type"]},
                        },
                        "required": ["min", "max"],
                    }
                    for key in continuous_schema["properties"].keys()
                },
            }
            bounds = cast(
                dict[str, Any],
                self.llm.generate(bounds_prompt, schema=bounds_schema, verbose=verbose),
            )

            continuous_schema = deepcopy(continuous_schema)
            for key, value in bounds.items():
                if key in continuous_schema["properties"]:
                    continuous_schema["properties"][key]["minimum"] = value["min"]
                    continuous_schema["properties"][key]["maximum"] = value["max"]

        disc = (
            self._sample_single(self.discrete_proposal_dist, discrete_schema, observed, verbose)
            if discrete_schema["properties"] else {}
        )
        cont = (
            self._sample_single(self.continuous_proposal_dist, continuous_schema, observed, verbose)
            if continuous_schema["properties"] else {}
        )
        return {**disc, **cont}, discrete_schema, continuous_schema

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        initial_sample, discrete_schema, continuous_schema = self._initialize_proposal_chain(
            schema, observed, verbose
        )
        has_discrete_features = bool(discrete_schema["properties"])
        has_continuous_features = bool(continuous_schema["properties"])

        samples = [initial_sample]

        chain_length = self.burn_in + n_samples * self.thinning
        for _ in tqdm(
            range(chain_length), disable=pbar is None, position=pbar, desc=f"Chain {pbar}", dynamic_ncols=True
        ):
            candidate = {}  # Prevent PossiblyUnboundVariable error from type checkers
            for _ in range(self.max_trials):  # Try up to max_trials times to generate a valid candidate
                candidate_discrete = {}
                if has_discrete_features:
                    candidate_discrete = self._sample_single(
                        self.discrete_proposal_dist, discrete_schema, observed, verbose
                    )

                candidate_continuous = {}
                if has_continuous_features:
                    candidate_continuous = self._sample_single(
                        self.continuous_proposal_dist, continuous_schema, observed, verbose
                    )

                candidate = {**candidate_discrete, **candidate_continuous}

                # If the candidate is the same as the previous sample, try again
                if all(samples[-1][k] == candidate[k] for k in candidate.keys()):
                    continue
                break

            if self.shuffle_variables:
                keys = list(candidate.keys())
                np.random.shuffle(keys)
                current = {k: samples[-1][k] for k in keys}
                candidate = {k: candidate[k] for k in keys}
            else:
                current = samples[-1]

            if np.random.choice([True, False]):
                options = [current, candidate]
            else:
                options = [candidate, current]

            try:
                if verbose:
                    print(f"Current sample: {current}, Candidate: {candidate}")

                if self._acceptance(options[0], options[1], observed, verbose=verbose):
                    samples.append(options[0])
                else:
                    samples.append(options[1])
            except Exception as e:
                if verbose:
                    print(f"Error during acceptance step: {e}. Rejecting candidate.")
                raise e

            if verbose:
                print(f"Generated {len(samples[self.burn_in + self.thinning::self.thinning][:n_samples])}/{n_samples} samples.")

        thinned_samples = samples[self.burn_in + self.thinning::self.thinning][:n_samples]
        return thinned_samples

    @abstractmethod
    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
    ) -> bool:
        pass


def default_barker_template(
    option1: dict[str, Any],
    option2: dict[str, Any],
    output_schema: dict[str, Any],
    observed: dict[str, Any] | None = None,
) -> str:
    template = ""
    if observed:
        template += (
            f"Given the observed features with these values: {json.dumps(observed)}, "
            "which of the following two options is more likely to be a valid data point? "
        )
    else:
        template += f"Which of the following two options is more likely to be a valid data point? "
    template += (
        f"Option 1: {json.dumps(option1)}. Option 2: {json.dumps(option2)}. "
        f"Respond in the format specified by this schema: {json.dumps(output_schema)}."
    )
    return template


class BarkerLLMPrior(MCMCLLMPrior):
    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None], str] = default_barker_template,
        burn_in: int = 10,
        thinning: int = 1,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="choice"),
    ):
        super().__init__(llm, template, burn_in, thinning, shuffle_variables, max_trials, manual_reasoning, reasoning_prompt)

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
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
            },
            "required": ["choice"]
        }

        if self.manual_reasoning:
            binary_schema["properties"] = {
                "reasoning": {"type": "string", "description": self.reasoning_prompt},
                **binary_schema["properties"],
            }
            binary_schema["required"] = ["reasoning"] + binary_schema["required"]

        input_str = self.template(option1, option2, binary_schema, observed)
        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("choice") == "Option 1"


def default_gambling_template(
    option1: dict[str, Any],
    option2: dict[str, Any],
    output_schema: dict[str, Any],
    bet_value: float,
    observed: dict[str, Any] | None = None,
) -> str:
    template = ""
    if observed:
        template += (
            "You will be presented with two sets of feature values for a data point, along with some observed "
            f"features with these values: {json.dumps(observed)}. One of these is real and the other is fake. "
        )
    else:
        template += (
            "You will be presented with two sets of feature values for a data point. One of these is real and the other is fake. "
        )
    template += (
        f"You have the opportunity to place a bet of ${bet_value} that Option 1 is more plausible, "
        "which will pay out $100 if you are correct. Your aim is to maximise profit. "
        f"Option 1 is {json.dumps(option1)} and Option 2 is {json.dumps(option2)}. "
        f"Respond with JSON that conforms to this schema: {json.dumps(output_schema)}."
    )
    return template


class GamblingLLMPrior(MCMCLLMPrior):
    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], float, dict[str, Any] | None], str] = default_gambling_template,
        burn_in: int = 10,
        thinning: int = 1,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="decision to place a bet or not"),
    ):
        super().__init__(llm, template, burn_in, thinning, shuffle_variables, max_trials, manual_reasoning, reasoning_prompt)

    def _acceptance(
        self,
        option1: dict[str, Any],
        option2: dict[str, Any],
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
            },
            "required": ["bet"]
        }

        if self.manual_reasoning:
            binary_schema["properties"] = {
                "reasoning": {"type": "string", "description": self.reasoning_prompt},
                **binary_schema["properties"],
            }
            binary_schema["required"] = ["reasoning"] + binary_schema["required"]

        bet_value = np.round(np.random.rand() * 100, 2)

        input_str = self.template(option1, option2, binary_schema, bet_value, observed)
        output = self.llm.generate(input_str, binary_schema, verbose=verbose)

        return type(output) is dict and output.get("bet") == "Place Bet"


class MetropolisWithinGibbsLLMPrior(MCMCLLMPrior):
    def __init__(
        self,
        llm: LLM,
        template: Callable[..., str],
        burn_in: int,
        thinning: int,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="decision"),
        block_size: int = 1,
        sweep: bool = False,
    ):
        super().__init__(llm, template, burn_in, thinning, shuffle_variables, max_trials, manual_reasoning, reasoning_prompt)
        self.block_size = block_size
        self.sweep = sweep
        if not (self.shuffle_variables or self.sweep):
            raise ValueError("Either shuffle_variables or sweep must be True.")

    def _sample_impl(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any] | None = None,
        verbose: bool = False,
        pbar: int | None = None,
    ) -> list[dict[str, Any]]:
        initial_sample, _, _ = self._initialize_proposal_chain(schema, observed, verbose)
        samples = [initial_sample]

        chain_length = self.burn_in + n_samples * self.thinning
        keys_pool = []
        for _ in tqdm(
            range(chain_length), disable=pbar is None, position=pbar, desc=f"Chain {pbar}", dynamic_ncols=True
        ):
            itr_observed = samples[-1].copy()
            keys = list(itr_observed.keys())
            if self.shuffle_variables:
                np.random.shuffle(keys)

            if not self.sweep:
                keys_to_discard = keys[-self.block_size:]
            else:
                if len(keys_pool) < self.block_size:
                    keys_pool = keys_pool + keys

                keys_to_discard = keys_pool[:self.block_size]
                keys_pool = keys_pool[self.block_size:]

            itr_discard = {k: itr_observed[k] for k in keys_to_discard}
            itr_observed = {k: itr_observed[k] for k in itr_observed if k not in keys_to_discard}

            itr_schema = {
                "type": "object",
                "properties": {
                    key: schema["properties"][key] for key in keys_to_discard
                },
                "required": keys_to_discard
            }

            all_observed = {**itr_observed, **(observed or {})}

            itr_discrete_schema, itr_continuous_schema = self.split_schema(itr_schema)
            has_discrete_features = bool(itr_discrete_schema["properties"])
            has_continuous_features = bool(itr_continuous_schema["properties"])

            candidate = {}  # Prevent PossiblyUnboundVariable error from type checkers
            for _ in range(self.max_trials):  # Try up to max_trials times to generate a valid candidate
                candidate_discrete = {}
                if has_discrete_features:
                    candidate_discrete = self._sample_single(
                        self.discrete_proposal_dist, itr_discrete_schema, all_observed, verbose
                    )

                candidate_continuous = {}
                if has_continuous_features:
                    candidate_continuous = self._sample_single(
                        self.continuous_proposal_dist, itr_continuous_schema, all_observed, verbose
                    )

                candidate = {**candidate_discrete, **candidate_continuous}

                # If the candidate is the same as the previous sample, try again
                if all(itr_discard[k] == candidate[k] for k in candidate.keys()):
                    continue
                break

            if np.random.choice([True, False]):
                options = [itr_discard, candidate]
            else:
                options = [candidate, itr_discard]

            try:
                if verbose:
                    print(f"Current sample: {itr_discard}, Candidate: {candidate}")

                if self._acceptance(options[0], options[1], all_observed, verbose=verbose):
                    new_sample = itr_observed | options[0]
                else:
                    new_sample = itr_observed | options[1]
                samples.append(new_sample)
            except Exception as e:
                if verbose:
                    print(f"Error during acceptance step: {e}. Rejecting candidate.")
                raise e

            if verbose:
                print(f"Generated {len(samples[self.burn_in + self.thinning::self.thinning][:n_samples])}/{n_samples} samples.")
                print(f"Current sample: {samples[-1]}")

        thinned_samples = samples[self.burn_in + self.thinning::self.thinning][:n_samples]
        return thinned_samples


class BarkerGibbsLLMPrior(MetropolisWithinGibbsLLMPrior, BarkerLLMPrior):
    """
    Block-Gibbs LLM prior whose per-block Metropolis update uses Barker-style LLM
    acceptance: at each step, the chain proposes a new block of variables from the
    proposal distributions and asks the LLM to pick the more plausible of the two
    options (current vs candidate).

    We reuse the `_sample_impl` from `MetropolisWithinGibbsLLMPrior` and the
    `_acceptance` from `BarkerLLMPrior` by python's multiple inheritance with MRO.
    """

    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any] | None], str] = default_barker_template,
        burn_in: int = 10,
        thinning: int = 1,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="choice"),
        block_size: int = 1,
        sweep: bool = False,
    ):
        super().__init__(
            llm, template, burn_in, thinning, shuffle_variables, max_trials, manual_reasoning, reasoning_prompt, block_size, sweep
        )


class GamblingGibbsLLMPrior(MetropolisWithinGibbsLLMPrior, GamblingLLMPrior):
    """
    Block-Gibbs LLM prior whose per-block Metropolis update uses gambling-style LLM
    acceptance: at each step, the chain proposes a new block of variables and asks
    the LLM whether to take a randomly-sized bet that the candidate is the real
    data point.

    We reuse the `_sample_impl` from `MetropolisWithinGibbsLLMPrior` and the
    `_acceptance` from `GamblingLLMPrior` by python's multiple inheritance with MRO.
    """

    def __init__(
        self,
        llm: LLM,
        template: Callable[[dict[str, Any], dict[str, Any], dict[str, Any], float, dict[str, Any] | None], str] = default_gambling_template,
        burn_in: int = 10,
        thinning: int = 1,
        shuffle_variables: bool = True,
        max_trials: int = 10,
        manual_reasoning: bool = False,
        reasoning_prompt: str = DEFAULT_REASONING_PROMPT.format(obj="decision to place a bet or not"),
        block_size: int = 1,
        sweep: bool = False,
    ):
        super().__init__(
            llm, template, burn_in, thinning, shuffle_variables, max_trials, manual_reasoning, reasoning_prompt, block_size, sweep
        )


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

    def sample_conditional(
        self,
        n_samples: int,
        schema: dict[str, Any],
        observed: dict[str, Any],
        verbose: bool = False,
    ) -> list[dict[str, Any]]:
        # Can't condition on observations — just draw from the marginal
        return self.sample(n_samples, schema, verbose)
