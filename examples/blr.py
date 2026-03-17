from argparse import ArgumentParser
from priorbot.data import load_dataset, split_dataset
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import LLMPrior, EmpiricalPrior, GibbsLLMPrior, BarkerLLMPrior, GamblingLLMPrior
from priorbot.skbayes import DPGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

if __name__ == "__main__":
    parser = ArgumentParser(description="Train a Bayesian Logistic Regression Model")
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset (in JSON format)")
    parser.add_argument("--model-name", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Name of the LLM model to use for the prior")
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the LLM API (if using a remote model)")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of samples to draw from the prior")
    parser.add_argument("--prior", type=str, choices=["direct", "gibbs", "barker", "gambling"], default="gambling")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    train_dataset, test_dataset = split_dataset(dataset, 0.8)

    system_prompt = f"You are an expert in the field of {dataset.domain}. Your top priority is to provide statisticians with the domain knowledge required to analyse their data. {dataset.description}"
    llm = OpenAICompatLLM(model_name=args.model_name, base_url=args.base_url, system_prompt=system_prompt)

    match args.prior:
        case "direct":
            base_prior = LLMPrior(llm=llm)
        case "gibbs":
            base_prior = GibbsLLMPrior(base_prior=LLMPrior(llm=llm), thinning=5)
        case "barker":
            base_prior = BarkerLLMPrior(llm=llm, thinning=5)
        case "gambling":
            base_prior = GamblingLLMPrior(llm=llm, thinning=5)
        case _:
            raise ValueError("Invalid prior type")

    full_schema = {
        "type": "object",
        "properties": {**dataset.feature_schema["properties"], **dataset.target_schema["properties"]},
        "required": dataset.feature_schema["required"] + dataset.target_schema["required"]
    }

    prior_samples = base_prior.sample(args.n_samples, schema=full_schema, verbose=True)
    prior = EmpiricalPrior(prior_samples)

    base_model = LogisticRegression(solver="liblinear")

    for alpha in [1.0, 10.0, 100.0]:
        model = DPGBClassifier(base_estimator=base_model, alpha=alpha)
        model.fit_informative(train_dataset, prior=prior)
        test_probs = model.predict_proba_dict(test_dataset.data)[:, 1]
        auc = roc_auc_score([data_point[model.target_name_] for data_point in test_dataset.data], test_probs)
        print(f"Alpha: {alpha}, Test AUC: {auc:.4f}")

