# PriorBot
This package contains utilities for eliciting Bayesian priors from large language models, and some tools for subsequently leveraging those priors for improved downstream analysis.

## Features
* Supports generating structured data from Local LLMs via `transformers` and `outlines`, and from remote LLMs using the `json_schema` facility of OpenAI-compatible APIs
* Naive direct sampling from LLMs
* Gibbs sampling from LLMs
* "Markov Chain Monte Carlo with People" sampling from LLMs
* A problem transformation approach for training Bayesian versions of `sklearn` classifiers using the LLM-elicited priors

## Examples
See the `examples/` directory for some basic demos. To run the `whisky.py` demo, you need to do this:

```bash
# Install the priorbot with remote LLMs enabled into the local python environment
uv sync --extra remote-llm

# Run the demo
OPENAI_API_KEY=yourkey uv run python examples/whisky.py --base-url=http://some.api.endpoint/v1 --model-name your-model/name
```

You can also pass the `--gibbs` or `--mcmc` flags to this example to explore different sampling approaches.
