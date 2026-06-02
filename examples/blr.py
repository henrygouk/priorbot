import os
from argparse import ArgumentParser

from priorbot.data import load_dataset, split_dataset
from priorbot.llm import OpenAICompatLLM
from priorbot.priors import (
    LLMPrior,
    EmpiricalPrior,
    GibbsLLMPrior,
    BarkerLLMPrior,
    GamblingLLMPrior,
    GamblingGibbsLLMPrior,
)
from priorbot.skbayes import DPGBClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Path to the dataset (in JSON format)")
    parser.add_argument(
        "--model-name",
        type=str,
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        help="Name of the LLM model to use for the prior",
    )
    parser.add_argument("--base-url", type=str, default=None, help="Base URL for the LLM API (if using a remote model)")
    parser.add_argument("--api-key", type=str, default="dummy", help="API key for the LLM API")
    parser.add_argument("--n-samples", type=int, default=128, help="Number of samples to draw from the prior")
    parser.add_argument(
        "--prior",
        type=str,
        choices=["direct", "gibbs", "barker", "gambling", "gambling_gibbs"],
        default="gambling",
    )
    parser.add_argument("--verbose", action="store_true", help="Whether to print verbose output during sampling")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_path)
    train_dataset, test_dataset = split_dataset(dataset, 0.8)

    feature_names = list(dataset.feature_schema.get("properties", {}).keys())
    categorical_features = [
        name
        for name, schema in dataset.feature_schema.get("properties", {}).items()
        if schema.get("type") == "string" or "enum" in schema
    ]
    numeric_features = [name for name in feature_names if name not in categorical_features]
    categorical_indices = [feature_names.index(name) for name in categorical_features]
    numeric_indices = [feature_names.index(name) for name in numeric_features]

    os.environ["OPENAI_API_KEY"] = args.api_key
    system_prompt = f"You are an expert in the field of {dataset.domain}. Your top priority is to provide statisticians with the domain knowledge required to analyse their data. {dataset.description}"
    llm = OpenAICompatLLM(model_name=args.model_name, base_url=args.base_url, system_prompt=system_prompt)

    match args.prior:
        case "direct":
            base_prior = LLMPrior(llm=llm)
        case "gibbs":
            base_prior = GibbsLLMPrior(llm_prior=LLMPrior(llm=llm), burn_in=10, thinning=5)
        case "barker":
            base_prior = BarkerLLMPrior(llm=llm, thinning=5)
        case "gambling":
            base_prior = GamblingLLMPrior(llm=llm, thinning=5)
        case "gambling_gibbs":
            base_prior = GamblingGibbsLLMPrior(
                llm=llm,
                burn_in=10,
                thinning=5,
            )
        case _:
            raise ValueError("Invalid prior type")

    full_schema = {
        "type": "object",
        "properties": {**dataset.feature_schema["properties"], **dataset.target_schema["properties"]},
        "required": dataset.feature_schema["required"] + dataset.target_schema["required"]
    }

    prior_samples = base_prior.sample(args.n_samples, schema=full_schema, verbose=args.verbose)
    prior = EmpiricalPrior(prior_samples)

    base_model = LogisticRegression(solver="liblinear")

    if categorical_features:
        preprocessor = ColumnTransformer(
            transformers=[
                ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_indices),
                ("numeric", "passthrough", numeric_indices),
            ],
            remainder="drop",
        )
        base_estimator = Pipeline(
            steps=[
                ("preprocess", preprocessor),
                ("model", base_model),
            ]
        )
    else:
        base_estimator = base_model

    for alpha in [1.0, 10.0, 100.0]:
        model = DPGBClassifier(base_estimator=base_estimator, alpha=alpha)
        model.fit_informative(train_dataset, prior=prior)
        test_probs = model.predict_proba_dict(test_dataset.data)[:, 1]
        auc = roc_auc_score([data_point[model.target_name_] for data_point in test_dataset.data], test_probs)
        print(f"Alpha: {alpha}, Test AUC: {auc:.4f}")

