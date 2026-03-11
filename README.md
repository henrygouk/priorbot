# PriorBot
This package contains utilities for eliciting Bayesian priors from large language models, and some tools for subsequently leveraging those priors for improved downstream analysis.

## Features
* Supports generating structured data from Local LLMs via `transformers` and `outlines`, and from remote LLMs using the `response_format` facility of OpenAI-compatible APIs
* Naive direct sampling from LLMs
* Gibbs sampling from LLMs
* "Markov Chain Monte Carlo with People" sampling from LLMs
* A modification of MCMC with People that uses a betting game
* A problem transformation approach for training Bayesian versions of `sklearn` classifiers using the LLM-elicited priors

## Examples
See the `examples/` directory for some basic demos. To run the `whisky.py` demo, you need to do this:

```bash
# Install the priorbot with remote LLMs enabled into the local python environment
uv sync --extra remote-llm

# Run the demo
OPENAI_API_KEY=yourkey uv run python examples/whisky.py --base-url=http://some.api.endpoint/v1 --model-name your-model/name
```

There is also a `--prior` argument that can be set to one of `direct`, `gibbs`, `barker`, or `gambling`. The `barker` option corresponds to MCMC with people.
