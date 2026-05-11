import mavenn

import pandas as pd
import numpy as np
from data import create_split

def load_gb1_data():
    """Load the GB1 dataset from the mavenn package."""
    data = mavenn.load_example_dataset("gb1")
    return data

def select_columns(data):
    """Select the relevant columns from the GB1 dataset."""
    return data[['x', 'y']]

def load_gb1_dataframe():
    """Load the GB1 dataset and return it as a pandas DataFrame with 'sequence' and 'binding_scores' columns."""
    data = load_gb1_data()
    df = pd.DataFrame(
        {
            "sequence": data["x"],
            "binding_scores": data["y"],
            "split": ["None"] * len(data["x"]),
        }
    )
    df = create_split.create_split(df)
    return df

