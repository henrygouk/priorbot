from abc import abstractmethod
import argparse
from data import Dataset, load_dataset, save_dataset
import json
import numpy as np
import outlines
from outlines.types.dsl import JsonSchema
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Optional

class LLMSampler:

    def __init__(self, max_tokens: int = 1024, system_prompt: str = ""):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    @abstractmethod
    def sample(self, prompt: str, output_type: JsonSchema) -> str:
        pass

    def sample_marginal(self, n_samples: int, marginal_schema: dict[str, Any], verbose: bool = False) -> list[dict[str, Any]]:
        samples = []
        input_str = format(f"Generate a data point that conforms to the following schema: {marginal_schema}")

        while len(samples) < n_samples:
            sample = self.sample(input_str, JsonSchema(marginal_schema))

            try:
                result = json.loads(sample)
                samples.append(result)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}. Sample: {sample}")
            except Exception as e:
                print(f"Unexpected error: {e}. Sample: {sample}")

            if verbose:
                print(f"Generated {len(samples)}/{n_samples} samples from the marginal distribution.")
        
        return samples

    def sample_conditional(self, marginals, conditional_schema: dict[str, Any], reasoning: bool=False, attempts: int=5, verbose: bool=False) -> tuple[list[dict[str, Any]], Optional[list[str]]]:
        samples = []
        prop_names = list(conditional_schema["properties"].keys())
        target_names = ", ".join(prop_names)

        if len(prop_names) == 1:
            field = "field"
            value = "value"
        else:
            field = "fields"
            value = "values"

        if reasoning:
            conditional_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": f"Step by step reasoning for the estimated {value} of the {target_names} {field}."},
                    **conditional_schema["properties"]
                },
                "required": conditional_schema["required"] + ["reasoning"]
            }

            if verbose:
                print(conditional_schema)

        for marginal in marginals:
            result = {}

            for _ in range(attempts):
                input_str = format(f"Given features with these values: {marginal}, estimate the {value} of {field}. Your response should conform to the following schema: {conditional_schema}")
                sample = self.sample(input_str, JsonSchema(conditional_schema))

                try:
                    result = json.loads(sample)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Sample: {sample}")
                    result = {}
                except Exception as e:
                    print(f"Unexpected error: {e}. Sample: {sample}")
                    result = {}

            samples.append(result)

            if verbose:
                print(f"Generated {len(samples)}/{len(marginals)} conditional samples.")
        
        if reasoning:
            reasoning_trace = [
                sample.pop("reasoning", None) for sample in samples
            ]
            return samples, reasoning_trace
        else:
            return samples, None
        
class LocalLLMSampler(LLMSampler):
    """
    A sampler that uses a local LLM to sample from a prior distribution.
    """

    def __init__(self, model_name: str, **kwargs):
        super().__init__(**kwargs)

        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = outlines.from_transformers(
            hf_model,
            hf_tokenizer
        )

        self.model = model
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer

    def sample(self, prompt: str, output_type: JsonSchema) -> str:
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt}
        ]

        input = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        output = self.model(input, output_type, max_new_tokens=self.max_tokens)
        return output

def sample_marginal_uniform(n_samples: int, marginal_schema: dict[str, Any], maxes: dict[str, float], mins: dict[str, float]) -> list[dict[str, Any]]:
    """
    Sample from a marginal distribution that does not make use of LLMs, but instead assumes uniform distributions.
    
    This is a marginal distribution over features that does not make an effort to match the real distribution of the data.
    It is designed such that the samples are easy to generate and the KL divergence between the heuristic marginal and the real marginal is finite.

    For numeric features, the heuristic marginal is a uniform distribution over the range of the feature.
    For categorical features, the heuristic marginal is a uniform distribution over the set of possible values.
    """
    samples = []
    marginal_schema = marginal_schema

    for _ in range(n_samples):
        sample = {}
        for key, value in marginal_schema["properties"].items():
            if value["type"] == "number":
                sample[key] = mins[key] + (maxes[key] - mins[key]) * np.random.rand()
            elif value["type"] == "string" and "enum" in value:
                sample[key] = np.random.choice(value["enum"])
            else:
                raise ValueError(f"Unsupported type {value.type} for key {key}")
        samples.append(sample)
    return samples

def sample_data(dataset: Dataset, model_name: str, num_samples: int, reasoning: bool, features: str, max_tokens: int, verbose: bool = False) -> Dataset:
    system_prompt =  f"You are an expert in the field of {dataset.domain}.\n"
    system_prompt += f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {dataset.description}\n"

    sampler = LocalLLMSampler(model_name=model_name, max_tokens=max_tokens, system_prompt=system_prompt)

    if verbose:
        print(f"Initialized LLMSampler with model: {model_name}")

    if features == "uniform":
        if verbose:
            print("Using uniform marginals.")
            print("Querying LLM for minimum and maximum values of numeric attributes.")
        
        maxes = {}
        mins = {}
        min_max_schema = JsonSchema({
            "type": "object",
            "properties": {
                "minimum": {"type": "number", "description": "The minimum value of the feature."},
                "maximum": {"type": "number", "description": "The maximum value of the feature."}
            },
            "required": ["minimum", "maximum"]
        })

        for feature_name, feature_schema in dataset.feature_schema["properties"].items():
            if feature_schema["type"] == "number":
                input_str = format(f"What are reasonable minimum and maximum values for the numeric feature '{feature_name}'?")
                response = sampler.sample(input_str, min_max_schema)
                try:
                    if verbose:
                        print(response)

                    response_json = json.loads(response)
                    mins[feature_name] = float(response_json["minimum"])
                    maxes[feature_name] = float(response_json["maximum"])
                except Exception as e:
                    if verbose:
                        print(f"Error parsing min/max for feature {feature_name}: {e}")

                    mins[feature_name] = 0.0
                    maxes[feature_name] = 1.0

        if verbose:
            print("Sampling marginals using uniform distribution within the estimated ranges.")
        
        marginals = sample_marginal_uniform(num_samples, dataset.feature_schema, maxes, mins)
        
        if verbose:
            print(f"Generated {len(marginals)} marginal samples using heuristic.")
    else:
        if verbose:
            print("Using LLM sampling for marginals.")

        marginals = sampler.sample_marginal(num_samples, dataset.feature_schema)
        
        if verbose:
            print(f"Generated {len(marginals)} marginal samples using LLM.")

    if reasoning:
        conditional_samples, reasons = sampler.sample_conditional(marginals, dataset.target_schema, reasoning, verbose=verbose)
    else:
        conditional_samples, reasons = sampler.sample_conditional(marginals, dataset.target_schema, reasoning, verbose=verbose)

    if verbose:
        print(f"Generated {len(conditional_samples)} conditional samples.")

    records = [
        marginal | conditional for marginal, conditional in zip(marginals, conditional_samples)
    ]

    output_data = Dataset(
        name=dataset.name,
        info=f"This data was generated using {model_name}. Original information: {dataset.info}",
        domain=dataset.domain,
        description=dataset.description,
        feature_schema=dataset.feature_schema,
        target_schema=dataset.target_schema,
        data=records,
        reasoning=reasons if reasoning else None
    )

    return output_data

def sample_data_cached(dataset: Dataset, model_name: str, num_samples: int, reasoning: bool, features: str, max_tokens: int, cache_path: str, verbose: bool = False):
    try:
        cached_data = load_dataset(cache_path)

        if verbose:
            print(f"Loaded cached data from {cache_path}")

        return cached_data
    except FileNotFoundError:
        if verbose:
            print(f"No cached data found at {cache_path}. Generating new samples.")

        output_data = sample_data(
            dataset=dataset,
            model_name=model_name,
            num_samples=num_samples,
            reasoning=reasoning,
            features=features,
            max_tokens=max_tokens,
            verbose=verbose
        )

        save_dataset(output_data, cache_path)

        if verbose:
            print(f"Saved generated samples to cache at {cache_path}")

        return output_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Prior Sampler")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the LLM model to use")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples to generate")
    parser.add_argument("--reasoning", action='store_true', help="Whether to include reasoning in conditional sampling")
    parser.add_argument("--features", choices=["uniform", "llm"], default="uniform", help="Feature sampling strategy")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate in each sample")
    parser.add_argument("--input-path", type=str, required=True, help="Path to input JSON file containing the dataset")
    parser.add_argument("--output-path", type=str, required=True, help="Path to output JSON file to save the samples")
    args = parser.parse_args()

    dataset = load_dataset(args.input_path)
    print(f"Loaded dataset from {args.input_path}")

    output_data = sample_data(
            dataset=dataset,
            model_name=args.model_name,
            num_samples=args.num_samples,
            reasoning=args.reasoning,
            features=args.features,
            max_tokens=args.max_tokens,
            verbose=True
        )

    save_dataset(output_data, args.output_path)
    print(f"Saved samples to {args.output_path}")
