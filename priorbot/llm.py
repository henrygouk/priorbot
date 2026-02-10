from abc import abstractmethod
from typing import Any, Optional
import json

class LLM:
    def __init__(self, model_name: str, **kwargs):
        self.model_name = model_name

    @abstractmethod
    def generate(self, prompt: str, output_type: None | dict[str, Any] = None) -> Optional[str | dict[str, Any]]:
        pass

class OpenAICompatLLM(LLM):
    """
    Use a (potentially) remote LLM accessed by an OpenAI-compatible API. This is should work with the official OpenAI
    and other compatible servers such as vLLM.

    This class depends on the `openai` library.

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

    response = llm.generate("Generate a data point that conforms to the following schema: {schema}", output_type=schema)
    ```
    """

    def __init__(self, model_name: str, base_url: str, system_prompt: str, **kwargs):
        """
        Initialise a client for an OpenAI-compatible API. The system prompt is used to set the behaviour of the LLM,
        and the base URL is used to specify the endpoint of the API.

        :param model_name: The name of the model to use (e.g. "meta-llama/Meta-Llama-3-8B-Instruct")
        :param base_url: The base URL of the OpenAI-compatible API (e.g. "http://localhost:8000/v1")
        :param system_prompt: The system prompt to use for the LLM (e.g. "You are a helpful assistant that generates data points conforming to a given schema.")
        """
        from openai import OpenAI

        super().__init__(model_name, **kwargs)
        self.base_url = base_url
        self.system_prompt = system_prompt
        self.client = OpenAI(base_url=base_url, **kwargs.get("openai_args", {}))

    def generate(self, prompt: str, output_type: None | dict[str, Any] = None) -> Optional[str | dict[str, Any]]:
        """
        Generate a response from the LLM given a prompt and an optional output type. The output type is used to specify
        the expected format of the response.

        :param prompt: The prompt to send to the LLM (e.g. "Generate a data point that conforms to the following schema: {schema}")
        :param output_type: The expected format of the response (e.g. a JSON schema dict). If None, the response is returned as a string.

        :return: The response from the LLM, either as a string or in the specified format (e.g. a dict conforming to the JSON schema).
        """
        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        if output_type is not None:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "output_schema",
                    "schema": output_type,
                }
            }

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat,
                response_format=response_format,
            )

            return json.loads(response.choices[0].message.content)
        else:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=chat,
            )

            return response.choices[0].message.content

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

    response = llm.generate("Generate a data point that conforms to the following schema: {schema}", output_type=schema)
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

    def generate(self, prompt: str, output_type: None | dict[str, Any] = None) -> Optional[str | dict[str, Any]]:
        """
        Generate a response from the LLM given a prompt and an optional output type. The output type is used to specify
        the expected format of the response.

        :param prompt: The prompt to send to the LLM (e.g. "Generate a data point that conforms to the following schema: {schema}")
        :param output_type: The expected format of the response (e.g. a JSON schema dict). If None, the response is returned as a string.

        :return: The response from the LLM, either as a string or in the specified format (e.g. a dict conforming to the JSON schema).
        """
        from outlines.types.dsl import JsonSchema

        chat = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        input_ids = self.hf_tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        output = self.model(input_ids, JsonSchema(output_type) if output_type else str)
        return output

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
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
            "age": {"type": "integer"}
        },
        "required": ["name", "age"]
    }

    response = llm.generate(f"Generate a data point that conforms to the following schema: {schema}", output_type=schema)
    print(response)
