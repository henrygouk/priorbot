from abc import abstractmethod
from typing import Any
import json


def _check_schema(data: dict[str, Any], schema: dict[str, Any]) -> None:
    """Raise an error if the data does not satisfy the schema."""
    props = schema.get("properties", {})
    for key, value in data.items():
        if key in props and props[key]["type"] in ["number", "integer"]:
            lo = props[key].get("minimum")
            hi = props[key].get("maximum")
            if (lo is not None and value < lo) or (hi is not None and value > hi):
                raise ValueError(f"Value {value} for key {key} is out of bounds for schema {schema}")

        if key in props and props[key]["type"] == "string":
            enum = props[key].get("enum")
            if enum is not None and value not in enum:
                raise ValueError(f"Value {value} for key {key} is not in enum {enum} for schema {schema}")

    required = schema.get("required", [])
    for key in required:
        if key not in data:
            raise ValueError(f"Key {key} is required but not present in data {data} for schema {schema}")


class LLM:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self, prompt: str, schema: None | dict[str, Any] = None, verbose: bool = False
    ) -> str | dict[str, Any]:
        pass


class OpenAICompatLLM(LLM):
    """
    Use a (potentially) remote LLM accessed by an OpenAI-compatible API. This is should work with the official OpenAI
    and other compatible servers such as vLLM.

    This class depends on the `openai` library.

    Automatically detects whether the served model supports chat templates. If it does, the chat.completions API is
    used; otherwise, falls back to the raw completions API.

    Example usage:
    ```python
    llm = OpenAICompatLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        base_url="http://localhost:8000/v1",
        system_prompt="You are a helpful assistant that generates data points conforming to a given schema.",
    )

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    response = llm.generate("Generate a data point that conforms to the following schema: {schema}", schema=schema)
    ```
    """

    def __init__(self, model_name: str, base_url: str, system_prompt: str, **kwargs):
        """
        Initialise a client for an OpenAI-compatible API. The system prompt is used to set the behaviour of the LLM,
        and the base URL is used to specify the endpoint of the API.

        :param model_name: The name of the model to use (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
        :param base_url: The base URL of the OpenAI-compatible API (e.g. "http://localhost:8000/v1")
        :param system_prompt: The system prompt to use for the LLM (e.g. "You are a helpful assistant that generates data points conforming to a given schema.")
        :param max_tokens: Maximum number of tokens to generate (default: 1024).
        :param temperature: The temperature to use for the LLM (default: 1.0).
        :param top_p: The top-p value to use for the LLM (default: 1.0).
        """
        from openai import OpenAI

        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=base_url, **kwargs.get("openai_args", {}))
        self.max_tokens = kwargs.get("max_tokens", 1024)
        self.temperature = kwargs.get("temperature", 1.0)
        self.top_p = kwargs.get("top_p", 1.0)
        self._use_chat_api: bool | None = None

    def _generate_chat(
        self, prompt: str, schema: None | dict[str, Any], verbose: bool
    ) -> str | dict[str, Any]:
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        if verbose:
            print(f"Chat prompt: ```\n{chat}\n```")

        kwargs = {
            "model": self.model_name,
            "messages": chat,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if schema is not None:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",
                    "schema": schema,
                }
            }
            if verbose:
                print(f"Response format: ```\n{kwargs['response_format']}\n```")

        response = self.client.chat.completions.create(**kwargs)
        if verbose:
            print(f"Response: ```\n{response}\n```")

        content = response.choices[0].message.content
        if schema is not None:
            return json.loads(content)
        return content
    
    def _generate_completion(
        self, prompt: str, schema: None | dict[str, Any], verbose: bool
    ) -> str | dict[str, Any]:
        prompt = f"{(self.system_prompt + '\n') if self.system_prompt else ''}{prompt}"
        if verbose:
            print(f"Completion prompt: ```\n{prompt}\n```")

        kwargs = {
            "model": self.model_name,
            "prompt": prompt,
            "max_tokens": self.max_tokens,  # openai completions default is 16
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if schema is not None:
            # For vllm >= 0.12.0; this might not work for other libraries (e.g., Ollama) or older versions of vllm
            kwargs["extra_body"] = {"structured_outputs": {"json": schema}}

        response = self.client.completions.create(**kwargs)
        if verbose:
            print(f"Response: ```\n{response}\n```")

        content = response.choices[0].text
        if schema is not None:
            return json.loads(content)
        return content

    def generate(
        self, prompt: str, schema: None | dict[str, Any] = None, verbose: bool = False, max_trials: int = 10
    ) -> str | dict[str, Any]:
        """
        Generate a response from the LLM given a prompt and an optional output type. The output type is used to specify
        the expected format of the response.

        :param prompt: The prompt to send to the LLM (e.g. "Generate a data point that conforms to the following schema: {schema}")
        :param schema: The expected format of the response (e.g. a JSON schema dict). If None, the response is returned as a string.
        :param verbose: Whether to print the prompt and response to the console.
        :param max_trials: The maximum number of trials to make if the response is not valid.

        :return: The response from the LLM, either as a string or in the specified format (e.g. a dict conforming to the JSON schema).
        """
        from openai import BadRequestError

        if self._use_chat_api is None:
            try:
                self._generate_chat(prompt, schema, verbose)
                self._use_chat_api = True
            except BadRequestError as e:
                if "chat template" in str(e).lower():
                    print("\nModel has no chat template — falling back to completions API.")
                    self._use_chat_api = False
            return self.generate(prompt, schema, verbose)

        for _ in range(max_trials):
            try:
                if self._use_chat_api:
                    content = self._generate_chat(prompt, schema, verbose)
                else:
                    content = self._generate_completion(prompt, schema, verbose)

                if schema is not None:
                    assert isinstance(content, dict)  # JSON-formatted response
                    _check_schema(content, schema)

            except Exception as e:
                print(f"Error during generation: {e}. Retrying...")
                continue

            return content

        raise RuntimeError(f"Failed to generate a valid response after {max_trials} trials.")


class OutlinesLocalLLM(LLM):
    """
    Use a local LLM via huggingface transformers and the outlines library. This is useful for running LLMs locally
    without needing to set up an API server, and also having appropriately constrained outputs.

    This class depends on the `outlines` and `transformers` libraries.

    Example usage:
    ```python
    llm = OutlinesLocalLLM(
        model_name="meta-llama/Meta-Llama-3-8B-Instruct",
        system_prompt="You are a helpful assistant that generates data points conforming to a given schema.",
    )

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    response = llm.generate("Generate a data point that conforms to the following schema: {schema}", schema=schema)
    """
    
    def __init__(self, model_name: str, system_prompt: str, **kwargs):
        """
        Initialise a local LLM using the outlines library. The system prompt is used to set the behaviour of the LLM.

        :param model_name: The name of the model to use (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
        :param system_prompt: The system prompt to use for the LLM (e.g. "You are a helpful assistant that generates data points conforming to a given schema.")
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import outlines

        super().__init__(model_name, **kwargs)
        self.system_prompt = system_prompt

        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        hf_tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

        model = outlines.from_transformers(
            hf_model,
            hf_tokenizer
        )

        self.model = model
        self.hf_model = hf_model
        self.hf_tokenizer = hf_tokenizer

    def generate(
        self, prompt: str, schema: None | dict[str, Any] = None, verbose: bool = False, max_trials: int = 10
    ) -> str | dict[str, Any]:
        """
        Generate a response from the LLM given a prompt and an optional output type. The output type is used to specify
        the expected format of the response.

        :param prompt: The prompt to send to the LLM (e.g. "Generate a data point that conforms to the following schema: {schema}")
        :param schema: The expected format of the response (e.g. a JSON schema dict). If None, the response is returned as a string.
        :param verbose: Whether to print the prompt and response to the console.
        :param max_trials: The maximum number of trials to make if the response is not valid.

        :return: The response from the LLM, either as a string or in the specified format (e.g. a dict conforming to the JSON schema).
        """
        from outlines.types.dsl import JsonSchema

        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        output = self.model(input_ids, JsonSchema(schema) if schema else str)
        if schema is not None:
            assert isinstance(output, dict)  # JSON-formatted response
            if not _check_schema(output, schema):
                if max_trials > 0:
                    return self.generate(prompt, schema, verbose, max_trials - 1)
                else:
                    raise RuntimeError(
                        f"Failed to generate a valid response after {max_trials} trials."
                    )
        return output


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=None)
    args = parser.parse_args()

    if args.base_url:
        llm = OpenAICompatLLM(
            model_name=args.model_name,
            base_url=args.base_url,
            system_prompt="You are a helpful assistant that generates data points conforming to a given schema.",
        )
    else:
        llm = OutlinesLocalLLM(
            model_name=args.model_name,
            system_prompt="You are a helpful assistant that generates data points conforming to a given schema.",
        )

    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
        "required": ["name", "age"],
    }

    response = llm.generate(
        f"Generate a data point that conforms to the following schema: {schema}",
        schema=schema,
        verbose=True,
    )
    print(response)
