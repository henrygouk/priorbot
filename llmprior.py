from abc import abstractmethod
from data import Dataset, load_dataset, save_dataset
import json
import outlines
from outlines.types.dsl import Alternatives, JsonSchema, String
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Any, Literal

class LLMSampler:

    def __init__(self, max_tokens: int = 1024, system_prompt: str = ""):
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    @abstractmethod
    def _sample(self, input: str, output_type: JsonSchema):
        pass

    def sample_marginal(self, n_samples: int, marginal_schema: dict[str, Any]):
        samples = []
        input_str = format(f"Generate a data point that conforms to the following schema: {marginal_schema}")

        while len(samples) < n_samples:
            sample = self._sample(input_str, JsonSchema(marginal_schema))

            try:
                result = json.loads(sample)
                samples.append(result)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}. Sample: {sample}")
            except Exception as e:
                print(f"Unexpected error: {e}. Sample: {sample}")

            print(f"Generated {len(samples)}/{n_samples} samples from the marginal distribution.")
        
        return samples

    def sample_conditional(self, marginals, conditional_schema: dict[str, Any], reasoning: bool=False, attempts: int=5):
        samples = []

        if reasoning:
            prop_names = list(conditional_schema["properties"].keys())
            target_names = ", ".join(prop_names)

            if len(prop_names) == 1:
                field = "field"
                value = "value"
            else:
                field = "fields"
                value = "values"

            conditional_schema = {
                "type": "object",
                "properties": {
                    "reasoning": {"type": "string", "description": f"Step by step reasoning for the estimated {value} of the {target_names} {field}."},
                    **conditional_schema["properties"]
                },
                "required": conditional_schema["required"] + ["reasoning"]
            }
            print(conditional_schema)

        for marginal in marginals:
            result = {}

            for _ in range(attempts):
                input_str = format(f"Given features with these values: {marginal}, generate the corresponding target. Your response should conform to the following schema: {conditional_schema}")
                sample = self._sample(input_str, JsonSchema(conditional_schema))

                try:
                    result = json.loads(sample)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}. Sample: {sample}")
                    result = {}
                except Exception as e:
                    print(f"Unexpected error: {e}. Sample: {sample}")
                    result = {}

            samples.append(result)
            print(f"Generated {len(samples)}/{len(marginals)} conditional samples.")
        
        if reasoning:
            reasoning = [
                sample.pop("reasoning", None) for sample in samples
            ]
            return samples, reasoning
        else:
            return samples
        
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

    def _sample(self, input: str, output_type: JsonSchema):
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": input}
        ]

        input = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        output = self.model(input, output_type, max_new_tokens=self.max_tokens)
        return output

def sample_marginal_heuristic(n_samples: int, marginal_schema: dict[str, Any]):
    """
    Sample from a heuristic marginal distribution that does not make use of LLMs, but instead of basics scipy distributions.
    
    This is a marginal distribution over features that does not make an effort to match the real distribution of the data.
    It is designed such that the samples are easy to generate and the KL divergence between the heuristic marginal and the real marginal is finite.

    For numeric features, the heuristic marginal is a uniform distribution over the range of the feature.
    For categorical features, the heuristic marginal is a uniform distribution over the set of possible values.
    """
    samples = []
    marginal_schema = marginal_schema.__dict__
    for _ in range(n_samples):
        sample = {}
        for key, value in marginal_schema["properties"].items():
            if value.type == "number":
                sample[key] = value.uniform_sample()
            elif value.type == "string" and "enum" in value:
                sample[key] = value.sample_from_enum()
            else:
                raise ValueError(f"Unsupported type {value.type} for key {key}")
        samples.append(sample)
    return samples

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LLM Prior Sampler")
    parser.add_argument("--model-name", type=str, required=True, help="Name of the LLM model to use")
    parser.add_argument("--num-samples", type=int, default=512, help="Number of samples to generate")
    parser.add_argument("--reasoning", action='store_true', help="Whether to include reasoning in conditional sampling")
    parser.add_argument("--conditional-only", action='store_true', help="Use heuristic sampling instead of LLM sampling when generating features.")
    parser.add_argument("--input-path", type=str, help="Path to input JSON file containing the dataset")
    parser.add_argument("--output-path", type=str, help="Path to output JSON file to save the samples")
    parser.add_argument("--max-tokens", type=int, default=1024, help="Maximum number of tokens to generate in each sample")
    args = parser.parse_args()

    dataset = load_dataset(args.input_path)
    print(f"Loaded dataset from {args.input_path}")

    nl = "\n"
    system_prompt =  f"You are an expert in the field of {dataset.domain}.\n"
    system_prompt += f"Your top priority is to provide statisticians with the domain knowedge required to analyse their data. {dataset.description}\n"

    sampler = LocalLLMSampler(model_name=args.model_name, max_tokens=args.max_tokens, system_prompt=system_prompt)
    print(f"Initialized LLMSampler with model: {args.model_name}")

    if args.conditional_only:
        print("Using heuristic sampling for marginals.")
        marginals = sample_marginal_heuristic(args.num_samples, dataset.feature_schema)
        print(f"Generated {len(marginals)} marginal samples using heuristic.")
    else:
        print("Using LLM sampling for marginals.")
        marginals = sampler.sample_marginal(args.num_samples, dataset.feature_schema)
        print(f"Generated {len(marginals)} marginal samples using LLM.")

    if args.reasoning:
        conditional_samples, reasons = sampler.sample_conditional(marginals, dataset.target_schema, args.reasoning)
    else:
        conditional_samples = sampler.sample_conditional(marginals, dataset.target_schema, args.reasoning)

    print(f"Generated {len(conditional_samples)} conditional samples.")

    records = [
        {**marginal, **conditional} for marginal, conditional in zip(marginals, conditional_samples)
    ]

    output_data = Dataset(
        name=dataset.name,
        info=f"This data was generated using {args.model_name}. Original information: {dataset.info}",
        domain=dataset.domain,
        description=dataset.description,
        feature_schema=dataset.feature_schema,
        target_schema=dataset.target_schema,
        data=records,
        reasoning=reasons if args.reasoning else None
    )

    save_dataset(output_data, args.output_path)
    print(f"Saved samples to {args.output_path}")