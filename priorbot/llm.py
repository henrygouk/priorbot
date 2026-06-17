from abc import abstractmethod
from typing import Any
import copy
import json
import re


def _strip_xgrammar_buggy_number_bounds(schema: Any) -> Any:
    """Workaround for an xgrammar bug in ``GenerateFloatRangeRegex``.

    When a JSON-schema number/integer has both ``minimum`` and ``maximum`` set
    and the closed range ``[minimum, maximum]`` overlaps the open interval
    ``(-1, 0)``, xgrammar generates a regex whose negative branch starts at
    ``-[1-9]`` and therefore excludes all ``-0.X`` values. We work around this
    by returning a deep copy of ``schema`` with those bounds removed from any
    affected numeric subschema. The original (bounded) schema is still used
    client-side by ``_check_json_schema`` to reject out-of-range samples.
    """

    def _overlaps_neg_unit(lo: Any, hi: Any) -> bool:
        if not isinstance(lo, (int, float)) or not isinstance(hi, (int, float)):
            return False
        return lo < 0.0 and hi > -1.0

    def _walk(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") in ("number", "integer") and _overlaps_neg_unit(
                node.get("minimum"), node.get("maximum")
            ):
                node.pop("minimum", None)
                node.pop("maximum", None)
            for value in node.values():
                _walk(value)
        elif isinstance(node, list):
            for item in node:
                _walk(item)

    sanitized = copy.deepcopy(schema)
    _walk(sanitized)
    return sanitized


def _check_schema(content: dict[str, Any] | str, schema: dict[str, Any] | str | list[str]) -> None:
    # Choice
    if isinstance(schema, list):
        if not isinstance(content, str):
            raise ValueError(f"Response is not a string: {content}")
        if content not in schema:
            raise ValueError(f"Response {content!r} is not in choice set {schema}")

    # Regex
    elif isinstance(schema, str):
        if not isinstance(content, str):
            raise ValueError(f"Response is not a string: {content}")
        if not re.search(schema, content):
            raise ValueError(f"Response {content} does not match regex schema {schema}")

    # JSON
    else:
        if not isinstance(content, dict):  # JSON-formatted response
            raise ValueError(f"Response is not a dictionary: {content}")

        props = schema.get("properties", {})
        for key, value in content.items():
            if key in props:
                _check_json_schema_value(value, props[key], key=key)

        required = schema.get("required", [])
        for key in required:
            if key not in content:
                raise ValueError(f"Key {key} is required but not present in data {content} for schema {schema}")


def _check_json_schema_value(value: Any, subschema: dict[str, Any], key: str) -> None:
    """Validate a single value against a JSON-schema subschema."""
    val_type = subschema.get("type")
    if val_type in ("number", "integer"):
        lo = subschema.get("minimum")
        hi = subschema.get("maximum")
        if (lo is not None and value < lo) or (hi is not None and value > hi):
            raise ValueError(f"Value {value} for key {key} is out of bounds for schema {subschema}")
    elif val_type == "string":
        enum = subschema.get("enum")
        if enum is not None and value not in enum:
            raise ValueError(f"Value {value} for key {key} is not in enum {enum} for schema {subschema}")
    elif val_type == "array":
        if not isinstance(value, list):
            raise ValueError(f"Value {value} for key {key} is not an array for schema {subschema}")
        min_items = subschema.get("minItems")
        max_items = subschema.get("maxItems")
        if min_items is not None and len(value) < min_items:
            raise ValueError(
                f"Array for key {key} has {len(value)} items, expected at least {min_items} "
                f"for schema {subschema}"
            )
        if max_items is not None and len(value) > max_items:
            raise ValueError(
                f"Array for key {key} has {len(value)} items, expected at most {max_items} "
                f"for schema {subschema}"
            )
        items_schema = subschema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                _check_json_schema_value(item, items_schema, key=f"{key}[{i}]")


class LLM:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(
        self,
        prompt: str,
        schema: None | dict[str, Any] | str | list[str] = None,
        verbose: bool = False,
        max_trials: int = 10,
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

    @staticmethod
    def _structured_outputs_kwargs(
        schema: dict[str, Any] | str | list[str], use_chat_api: bool
    ) -> dict[str, Any]:
        """Build the vLLM ``extra_body``/``response_format`` payload for ``schema``.

        - ``list[str]``: constrain the output to one of the given strings via
          vLLM's ``structured_outputs.choice`` (returns a raw string).
        - ``str``: treat as a regex pattern.
        - ``dict``: treat as a JSON schema.
        """
        if isinstance(schema, dict):
            schema = _strip_xgrammar_buggy_number_bounds(schema)

            if use_chat_api:
                # JSON schema is best expressed through response_format on chat.
                return {
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "output_schema",
                            "schema": schema,
                        },
                    }
                }
            else:
                return {"extra_body": {"structured_outputs": {"json": schema}}}

        elif isinstance(schema, list):
            return {"extra_body": {"structured_outputs": {"choice": schema}}}
        elif isinstance(schema, str):
            return {"extra_body": {"structured_outputs": {"regex": schema}}}
        else:
            raise ValueError(f"Invalid schema type: {type(schema)}")

    def _prepare_kwargs(
        self,
        prompt: str,
        schema: None | dict[str, Any] | str | list[str],
        use_chat_api: bool,
        verbose: bool,
    ) -> dict[str, Any]:
        kwargs = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

        if schema is not None:
            # For vllm >= 0.12.0; this might not work for other libraries
            # (e.g., Ollama) or older versions of vllm.
            kwargs.update(self._structured_outputs_kwargs(schema, use_chat_api))

        if use_chat_api:
            chat = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            if verbose:
                print(f"Chat prompt: ```\n{chat}\n```")
            kwargs["messages"] = chat
        else:
            kwargs["prompt"] = (
                f"{(self.system_prompt + '\n') if self.system_prompt else ''}{prompt}"
            )
            if verbose:
                print(f"Completion prompt: ```\n{kwargs['prompt']}\n```")
        return kwargs

    def _generate(
        self,
        prompt: str,
        schema: None | dict[str, Any] | str | list[str],
        use_chat_api: bool,
        verbose: bool,
    ) -> str | dict[str, Any]:
        kwargs = self._prepare_kwargs(prompt, schema, use_chat_api, verbose)

        if use_chat_api:
            response = self.client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content
        else:
            response = self.client.completions.create(**kwargs)
            content = response.choices[0].text

        if verbose:
            print(f"Response: ```\n{response}\n```")

        if isinstance(schema, dict):
            return json.loads(content)
        return content

    def generate(
        self,
        prompt: str,
        schema: None | dict[str, Any] | str | list[str] = None,
        verbose: bool = False,
        max_trials: int = 10,
    ) -> str | dict[str, Any]:
        """
        Generate a response from the LLM given a prompt and an optional output type. The output type is used to specify
        the expected format of the response.

        :param prompt: The prompt to send to the LLM (e.g. "Generate a data point that conforms to the following schema: {schema}")
        :param schema: The expected format of the response. ``None`` returns a free-form string. A ``dict`` is treated
            as a JSON schema (returns a parsed dict). A ``str`` is treated as a regex pattern. A ``list[str]`` is
            treated as a constrained "choice" set and returns one of the listed strings verbatim.
        :param verbose: Whether to print the prompt and response to the console.
        :param max_trials: The maximum number of trials to make if the response is not valid.

        :return: The response from the LLM, either as a string or in the specified format (e.g. a dict conforming to the JSON schema).
        """
        from openai import BadRequestError

        for _ in range(max_trials):
            try:
                content = None
                if self._use_chat_api is None:
                    try:
                        content = self._generate(prompt, schema, use_chat_api=True, verbose=verbose)
                        self._use_chat_api = True
                    except BadRequestError as e:
                        if "chat template" in str(e).lower():
                            print("\nModel has no chat template — falling back to completions API.")
                            self._use_chat_api = False
                        else:
                            raise

                assert self._use_chat_api is not None
                if content is None:
                    content = self._generate(prompt, schema, use_chat_api=self._use_chat_api, verbose=verbose)

                if schema is not None:
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
        self,
        prompt: str,
        schema: None | dict[str, Any] | str | list[str] = None,
        verbose: bool = False,
        max_trials: int = 10,
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
        if isinstance(schema, list) or isinstance(schema, str):
            raise ValueError(f"Schema type {type(schema)} is not supported by OutlinesLocalLLM.")

        from outlines.types.dsl import JsonSchema

        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

        for _ in range(max_trials):
            try:
                output = self.model(input_ids, JsonSchema(schema) if schema else str)

                if schema is not None:
                    _check_schema(output, schema)

            except Exception as e:
                print(f"Error during generation: {e}. Retrying...")
                continue

            return output

        raise RuntimeError(f"Failed to generate a valid response after {max_trials} trials.")


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
