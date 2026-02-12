from priorbot.llm import OpenAICompatLLM
from priorbot.priors import LLMPrior, GibbsSamplingPrior, MCMCLLMPrior

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--gibbs", action="store_true", help="Whether to use Gibbs sampling.")
    parser.add_argument("--mcmc", action="store_true", help="Whether to use MCMC with People sampling.")
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
                "enum": ["12", "15", "18", "21", "25"]
            },
            "Region": {
                "type": "string",
                "enum": ["Speyside", "Islay", "Highland", "Lowland", "Campbeltown"]
            }
        }
    }

    system_prompt = "You are a data scientist and whisky connoisseur ready to answer some questions about the range of whisky available at Tesco."

    if args.mcmc:
        prior = MCMCLLMPrior(llm=OpenAICompatLLM(
                base_url=args.base_url,
                model_name=args.model_name, 
                system_prompt=system_prompt
            )
        )
    else:
        prior = LLMPrior(llm=OpenAICompatLLM(
            base_url=args.base_url,
            model_name=args.model_name,
            system_prompt=system_prompt
        ))

        if args.gibbs:
            prior = GibbsSamplingPrior(base_prior=prior)

    samples = prior.sample(10, schema, verbose=True)
    print(samples)
