from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerLLMPrior, GibbsLLMPrior, LLMPrior, GamblingLLMPrior, UniformPrior

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--prior", type=str, choices=["uniform", "direct", "gibbs", "barker", "gambling", "gambling_gibbs"], default="gambling")
    args = parser.parse_args()

    schema = {
        "type": "object",
        "properties": {
            "Distillery": {
                "type": "string",
                "enum": ["Glenfiddich", "Macallan", "Lagavulin", "Laphroaig", "Ardbeg"]
            },
            "Age (years)": {
                "type": "integer",
                "maximum": 100,
                "minimum": 1,
            },
            "ABV (%)": {
                "type": "number",
                "maximum": 100,
                "minimum": 0,
            },
            "Region": {
                "type": "string",
                "enum": ["Speyside", "Islay"]
            }
        },
        "required": ["Distillery", "Age (years)", "ABV (%)", "Region"]
    }

    system_prompt = "You are a data scientist and whisky expert tasked with investigating purchase records from a popular Scottish supermarket."

    llm = OpenAICompatLLM(
        base_url=args.base_url,
        model_name=args.model_name,
        system_prompt=system_prompt,
        temperature=1.0,
        top_p=1.0,
        max_tokens=64,
    )

    match args.prior:
        case "uniform":
            prior = UniformPrior()
        case "direct":
            prior = LLMPrior(llm=llm)
        case "gibbs":
            prior = GibbsLLMPrior(base_prior=LLMPrior(llm=llm), burn_in=10, thinning=5)
        case "barker":
            prior = BarkerLLMPrior(llm=llm, thinning=5)
        case "gambling":
            prior = GamblingLLMPrior(llm=llm, thinning=5)
        case "gambling_gibbs":
            prior = GibbsLLMPrior(
                base_prior=GamblingLLMPrior(llm=llm, burn_in=0, thinning=1),
                burn_in=10,
                thinning=5,
            )
        case _:
            raise ValueError("Invalid prior type")

    # samples = prior.sample(100, schema, verbose=True, pbar=True)
    samples = prior.sample_parallel(5, [schema] * 10, verbose=True, pbar=True)
    print(samples)
