from dataclasses import dataclass
from enum import Enum
import json
from pydantic import BaseModel, Field
from typing import Any, List, Tuple
import numpy as np
import pandas as pd

class Dataset(BaseModel):
    """
    Represents a dataset with feature and target schemas.
    """
    name: str = Field(..., description="Name of the dataset.")
    info: str = Field(..., description="Additional information about the dataset, such as who collected it and copyright information.")
    domain: str = Field(..., description="Domain of the dataset, e.g., 'healthcare', 'finance'.")
    description: str = Field(..., description="Description of the dataset.")
    feature_schema: dict[str, Any] = Field(..., description="Schema for the features of the dataset.")
    target_schema: dict[str, Any] = Field(..., description="Schema for the target variable of the dataset.")
    data: List[dict[str, Any]] = Field(..., description="List of data points in the dataset.")
    reasoning: List[str] | None = Field(None, description="Optional reasoning (nautral language) for why the given target was assigned, if available.")

def load_dataset(file_path: str) -> Dataset:
    """
    Load a dataset from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing the dataset.

    Returns:
        Dataset: An instance of the Dataset class.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return Dataset(**data)

def save_dataset(dataset: Dataset, file_path: str):
    """
    Save a dataset to a JSON file.

    Args:
        dataset (Dataset): The dataset to save.
        file_path (str): Path to the JSON file where the dataset will be saved.
    """
    with open(file_path, 'w') as f:
        f.write(dataset.model_dump_json(indent=2))

@dataclass
class ARFFAttribute:
    name: str
    description: str
    dtype: str
    values: List[str] | None

@dataclass
class ARFFMetaData:
    name: str
    description: str
    field: str
    features: List[ARFFAttribute]
    target: ARFFAttribute

def load_arff(data_path: str) -> Tuple[ARFFMetaData, np.ndarray, np.ndarray]:
    import arff
    records = arff.load(open(data_path, 'r'))

    data = np.array(records["data"])
    X = data[:,:-1]
    y = data[:,-1]

    name_parts = records["relation"].split('#')
    field = name_parts[0].strip()
    rel_name = name_parts[1].strip()

    features = []

    for i, (schema, attrib_type) in enumerate(records["attributes"][:-1]):
        parts = schema.split(':')
        name = parts[0].strip()
        description = parts[1].strip()

        if isinstance(attrib_type, list):
            X[:, i] = [attrib_type.index(v) for v in X[:, i]]
            features.append(ARFFAttribute(name=name, description=description, dtype="str", values=attrib_type))
        else:
            features.append(ARFFAttribute(name=name, description=description, dtype="float", values=None))

    target_parts = records["attributes"][-1][0].split(':')
    target_name = target_parts[0].strip()
    target_description = target_parts[1].strip()

    target = ARFFAttribute(name=target_name, description=target_description, dtype="str", values=np.unique(y).tolist())

    meta_data = ARFFMetaData(name=rel_name, description=records["description"], field=field, features=features, target=target)

    # Convert X to float
    X = X.astype(float)

    # Convert y from string class names to integers
    y = np.array([target.values.index(y_i) for y_i in y])

    return meta_data, X, y

def convert_arff(input_path: str, output_path: str):
    """
    Convert an ARFF dataset to a JSON dataset.

    Args:
        input_path (str): Path to the input ARFF file.
        output_path (str): Path to save the output JSON file.
    """
    meta_data, X, y = load_arff(input_path)

    feature_schema = {
        "type": "object",
        "properties": {},
        "required": [feature.name for feature in meta_data.features]
    }

    for feature in meta_data.features:
        if feature.values is not None:
            feature_schema["properties"][feature.name] = {
                "type": "string",
                "description": feature.description,
                "enum": feature.values
            }
        else:
            feature_schema["properties"][feature.name] = {
                "type": "number",
                "description": feature.description
            }

    target_schema = {
        "type": "object",
        "properties": {
            meta_data.target.name: {
                "type": "string",
                "description": meta_data.target.description,
                "enum": meta_data.target.values
            }
        },
        "required": [meta_data.target.name]
    }

    data = []
    for i in range(len(X)):
        data_point = {}

        for j, feature in enumerate(meta_data.features):
            if feature.values is not None:
                data_point[feature.name] = feature.values[int(X[i][j])]
            else:
                data_point[feature.name] = float(X[i][j])

        data_point[meta_data.target.name] = meta_data.target.values[int(y[i])]

        data.append(data_point)

    dataset = Dataset(domain=meta_data.field, description=meta_data.description, feature_schema=feature_schema, target_schema=target_schema, data=data, name=meta_data.name, info="", reasoning=None)

    save_dataset(dataset, output_path)

def convert_csv(input_path: str, output_path: str):
    """
    Convert a CSV dataset to a JSON dataset. Leaves the metadata fields empty.

    Args:
        input_path (str): Path to the input CSV file.
        output_path (str): Path to save the output JSON file.
    """
    df = pd.read_csv(input_path)
    
    feature_schema = {
        "type": "object",
        "properties": {},
        "required": list(df.columns[:-1])
    }

    for col in df.columns[:-1]:
        # Do some basic type inference
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_schema["properties"][col] = {
                "type": "number",
                "description": ""
            }
        else:
            feature_schema["properties"][col] = {
                "type": "string",
                "description": "",
                "enum": df[col].dropna().unique().tolist()
            }

    target_schema = {
        "type": "object",
        "properties": {
            df.columns[-1]: {
                "type": "string",
                "description": "",
                "enum": df[df.columns[-1]].dropna().unique().tolist()
            }
        },
        "required": [df.columns[-1]]
    }
    
    data = df.to_dict(orient='records')
    dataset = Dataset(
        name="",
        info="",
        domain="",
        description="",
        feature_schema=feature_schema,
        target_schema=target_schema,
        data=data,
        reasoning=None
    )
    
    save_dataset(dataset, output_path)
    
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Load and save datasets.")
    parser.add_argument("--input-path", type=str, help="Path to the input dataset file.")
    parser.add_argument("--output-path", type=str, help="Path to save the output JSON dataset file.")
    parser.add_argument("--format", type=str, choices=["arff", "csv"], default="arff", help="Format of the input dataset file.")
    args = parser.parse_args()
    
    if args.format == "arff":
        convert_arff(args.input_path, args.output_path)
    elif args.format == "csv":
        # Placeholder for CSV conversion logic
        pass
    else:
        raise ValueError(f"Unsupported format: {args.format}")
