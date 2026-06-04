import sys
import os
sys.path.insert(0, '/Users/meier/Documents/MasterThesis')
os.chdir('/Users/meier/Documents/MasterThesis')

import numpy as np
import pandas as pd
import data.create_split as create_split
import data.tfbind8_data as tfbind8_data
import data.gb1_data as gb1_data

from typing import List, Dict

def decode_sequence(encoded: np.ndarray, alphabet: List[str]) -> str:
    return "".join(alphabet[int(token)] for token in encoded)


def encode_sequence(sequence: str, token_to_idx: Dict[str, int]) -> np.ndarray:
    return np.array([token_to_idx[token] for token in sequence], dtype=np.int32)

def one_hot_encode_sequence(sequence: np.array, num_tokens:int) -> np.ndarray:
    one_hot = np.zeros((len(sequence), num_tokens), dtype=np.float32)
    for i, token in enumerate(sequence):
        one_hot[i, int(token)] = 1.0
    return one_hot.flatten()

def one_hot_decode_sequence(encoded: np.ndarray, alphabet: List[str]) -> np.ndarray:
    num_tokens = len(alphabet)
    sequence_length = len(encoded) // num_tokens
    one_hot = encoded.reshape((sequence_length, num_tokens))
    decoded_sequence = "".join(alphabet[np.argmax(row)] for row in one_hot)
    return decoded_sequence

def load_data(name: str):
    '''Loads the data for the specified dataset name.'''
    if name == "tfbind8":
        transcription_factor = "SIX6_REF_R1"
        return tfbind8_data.load_tfbind8(transcription_factor=transcription_factor)
    if name == "gb1":
        return gb1_data.load_gb1_dataframe()
    else:
        raise ValueError(f"Unknown dataset: {name}")
    
def build_tfbind8_dataframe(x: np.ndarray, y: np.ndarray, alphabet: List[str]) -> pd.DataFrame:
    '''Builds a DataFrame for the TFBind8 dataset. 
    The DataFrame has columns "sequence", "binding_scores", and "split".
    Calls create_split to assign split labels.'''
    y = np.asarray(y).reshape(-1).astype(np.float32)
    sequences = [decode_sequence(row, alphabet) for row in x]

    df = pd.DataFrame(
        {
            "sequence": sequences,
            "binding_scores": y,
            "split": ["None"] * len(sequences),
        }
    )
    df = create_split.create_split(df)
    df["sequence"] = df["sequence"].str.upper()
    return df