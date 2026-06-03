from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable
import json
import numpy as np

from .llm import LLM



class Proposal(ABC):
    @abstractmethod
    def sample(self, n_samples: int) -> list[Any]:
        """Sample ``n_samples`` values for a single covariate."""
        pass


class UniformProposal(Proposal):
    def __init__(
        self,
        *,
        enum: list[Any] | None = None,
        minimum: float | int | None = None,
        maximum: float | int | None = None,
        decimals: int | None = None,
    ):
        self.enum = enum
        self.minimum = minimum
        self.maximum = maximum
        self.decimals = decimals

    def sample(self, n_samples: int) -> list[Any]:
        if self.enum is not None:
            return np.random.choice(self.enum, size=n_samples).tolist()

        if self.minimum is None or self.maximum is None:
            raise ValueError("Minimum and maximum must be specified for uniform numeric proposals.")

        samples = np.random.uniform(self.minimum, self.maximum, size=n_samples)
        if self.decimals is not None:
            samples = np.round(samples, decimals=self.decimals)
        return samples.tolist()


class GaussianProposal(Proposal):
    def __init__(self, mean: float, std: float, decimals: int | None = None):
        self.mean = mean
        self.std = std
        self.decimals = decimals

    def sample(self, n_samples: int) -> list[Any]:
        samples = np.random.normal(self.mean, self.std, size=n_samples)
        if self.decimals is not None:
            samples = np.round(samples, decimals=self.decimals)
        return samples.tolist()


class LogNormalProposal(Proposal):
    def __init__(self, log_mean: float, log_std: float, decimals: int | None = None):
        self.log_mean = log_mean
        self.log_std = log_std
        self.decimals = decimals

    def sample(self, n_samples: int) -> list[Any]:
        samples = np.random.lognormal(mean=self.log_mean, sigma=self.log_std, size=n_samples)
        if self.decimals is not None:
            samples = np.round(samples, decimals=self.decimals)
        return samples.tolist()


class ProposalGenerator(ABC):
    @abstractmethod
    def generate(self, schema: dict[str, Any]) -> dict[str, Proposal]:
        """Return a proposal distribution per covariate name in the schema."""
        pass


class UniformProposalGenerator(ProposalGenerator):
    def generate(self, schema: dict[str, Any]) -> dict[str, Proposal]:
        proposals: dict[str, Proposal] = {}
        for key, spec in schema.get("properties", {}).items():
            if "enum" in spec:
                proposals[key] = UniformProposal(enum=spec["enum"])
                continue

            val_type = spec.get("type")
            if val_type in ("integer", "number"):
                if spec.get("minimum") is None or spec.get("maximum") is None:
                    raise ValueError("Minimum and maximum must be specified for uniform numeric proposals.")
                decimals = 0 if val_type == "integer" else spec.get("decimals")
                proposals[key] = UniformProposal(
                    minimum=spec["minimum"],
                    maximum=spec["maximum"],
                    decimals=decimals,
                )
                continue

            raise ValueError(f"Unsupported type {val_type} for uniform proposal")

        return proposals


def default_llm_proposal_template(
    feature_name: str,
    feature_schema: dict[str, Any],
    output_schema: dict[str, Any],
) -> str:
    return (
        "Given the following feature schema, choose an appropriate univariate proposal distribution (including parameters) for MCMC algorithms. "
        "Your priority is not to be exact, but to be reasonable and to capture the scale of the feature. Samples from the proposal will be rounded according to the decimals field. "
        f"Feature name: {feature_name}. "
        f"Feature schema: {json.dumps(feature_schema)}. "
        "Respond in JSON that conforms to this schema: "
        f"{json.dumps(output_schema)}."
    )


class LLMProposalGenerator(ProposalGenerator):
    def __init__(
        self,
        llm: LLM,
        template: Callable[[str, dict[str, Any], dict[str, Any]], str] = default_llm_proposal_template,
        verbose: bool = False,
    ):
        self.llm = llm
        self.template = template
        self.verbose = verbose

    @staticmethod
    def _serialize_proposal(proposal: Proposal) -> dict[str, Any]:
        if isinstance(proposal, UniformProposal):
            return {
                "type": "uniform",
                "enum": proposal.enum,
                "minimum": proposal.minimum,
                "maximum": proposal.maximum,
                "decimals": proposal.decimals,
            }
        if isinstance(proposal, GaussianProposal):
            return {
                "type": "gaussian",
                "mean": proposal.mean,
                "std": proposal.std,
                "decimals": proposal.decimals,
            }
        if isinstance(proposal, LogNormalProposal):
            return {
                "type": "lognormal",
                "log_mean": proposal.log_mean,
                "log_std": proposal.log_std,
                "decimals": proposal.decimals,
            }
        return {"type": type(proposal).__name__}

    def _output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "proposal_type": {
                    "type": "string",
                    "enum": ["uniform", "gaussian", "lognormal"],
                },
                "params": {
                    "type": "object",
                    "properties": {
                        "minimum": {"type": "number"},
                        "maximum": {"type": "number"},
                        "mean": {"type": "number"},
                        "std": {"type": "number"},
                        "log_mean": {"type": "number"},
                        "log_std": {"type": "number"},
                        "decimals": {"type": "integer"},
                    },
                    "required": ["decimals"],
                },
            },
            "required": ["proposal_type", "params"],
        }

    def generate(self, schema: dict[str, Any]) -> dict[str, Proposal]:
        proposals: dict[str, Proposal] = {}
        output_schema = self._output_schema()

        for key, spec in schema.get("properties", {}).items():
            if "enum" in spec:
                proposals[key] = UniformProposal(enum=spec["enum"])
                continue

            if spec.get("type") not in ("number", "integer"):
                raise ValueError(f"No proposal generated for key {key}.")

            feature_spec = dict(spec)
            feature_spec.pop("minimum", None)
            feature_spec.pop("maximum", None)
            prompt = self.template(key, feature_spec, output_schema)
            response = self.llm.generate(prompt, schema=output_schema, verbose=self.verbose)
            if not isinstance(response, dict):
                raise ValueError("LLM proposal generator returned invalid output.")

            proposal_type = response.get("proposal_type")
            params = response.get("params", {})
            decimals = params.get("decimals")
            if decimals is None or spec.get("type") == "integer":
                decimals = 0

            if proposal_type == "uniform":
                proposals[key] = UniformProposal(
                    minimum=params.get("minimum"),
                    maximum=params.get("maximum"),
                    decimals=decimals,
                )
            elif proposal_type == "gaussian":
                proposals[key] = GaussianProposal(
                    mean=params["mean"],
                    std=params["std"],
                    decimals=decimals,
                )
            elif proposal_type == "lognormal":
                proposals[key] = LogNormalProposal(
                    log_mean=params["log_mean"],
                    log_std=params["log_std"],
                    decimals=decimals,
                )
            else:
                raise ValueError(f"Unsupported proposal type {proposal_type} for key {key}.")

        if self.verbose:
            print("LLM proposals:")
            print(
                json.dumps(
                    {
                        key: self._serialize_proposal(value)
                        for key, value in proposals.items()
                    },
                    indent=2,
                )
            )
        return proposals
