from priorbot.llm import OpenAICompatLLM
from priorbot.priors import BarkerLLMPrior, GibbsLLMPrior, LLMPrior, GamblingLLMPrior

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--prior", type=str, choices=["direct", "gibbs", "barker", "gambling", "gambling_gibbs"], default="gambling")
    args = parser.parse_args()

    schema = {
        "type": "object",
        "properties": {
            "Distillery": {
                "type": "string",
                "enum": ["Glenfiddich", "Macallan", "Lagavulin", "Laphroaig", "Ardbeg"]
            },
            "Age": {
                "type": "string",
                "enum": ["10", "12", "16", "18", "25"]
            },
            "Region": {
                "type": "string",
                "enum": ["Speyside", "Islay"]
            }
        }
    }

    system_prompt = "You are a data scientist and whisky expert tasked with investigating purchase records from a popular Scottish supermarket."

    llm = OpenAICompatLLM(base_url=args.base_url, model_name=args.model_name, system_prompt=system_prompt)

    match args.prior:
        case "direct":
            prior = LLMPrior(llm=llm)
        case "gibbs":
            prior = GibbsLLMPrior(base_prior=LLMPrior(llm=llm), thinning=5)
        case "barker":
            prior = BarkerLLMPrior(llm=llm, thinning=5)
        case "gambling":
            prior = GamblingLLMPrior(llm=llm, thinning=5)
        case "gambling_gibbs":
            prior = GibbsLLMPrior(base_prior=GamblingLLMPrior(llm=llm, burn_in=0, thinning=1), thinning=5)
        case _:
            raise ValueError("Invalid prior type")

    samples = prior.sample(10, schema, verbose=True)
    print(samples)
