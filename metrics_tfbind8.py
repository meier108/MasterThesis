''''This file contains the code for the metrics used in the evaluation of the models.
It contains the following metrics:
- Spearman's rank correlation coefficient: measures the monotonic relationship between two variables. It is
    calculated as the Pearson correlation coefficient between the ranked variables.
- MSE: measures the average squared difference between the predicted and true values. It is calculated as the mean of the squared differences between the predicted and true values.
- Bias: measures the average difference between the predicted and true values. It is calculated as the mean of the differences between the predicted and true values.


If the main loop is executed following is happening:
1. The oracle is initialized with the dataset.
2. A train slplit is created. 
3. RF and MLP are trained on the train split. 
4. All sequences in the remaining TFBind8 dataset are evaluated with the trained models.
5. The scores of the trained models are compared to the oracle scores using the metrics defined in this file.
-> The calculated results are stored in the metrics.csv file.'''

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
from typing import Dict

from models import random_forest, mlp

from assets.data_ops import load_data, build_tfbind8_dataframe, encode_sequence, one_hot_encode_sequence
from assets.compute_metrics import min_hamming_distance
from data.create_split import create_split
from tqdm import tqdm



def calculate_spearmanr(y_true, y_pred):
    """Calculate Spearman's rank correlation coefficient."""
    return spearmanr(y_true, y_pred).correlation

def calculate_mse(y_true, y_pred):    
    """Calculate Mean Squared Error."""
    return mean_squared_error(y_true, y_pred)

def calculate_bias(y_true, y_pred):
    """Calculate Bias."""
    return np.mean(y_pred - y_true)

def create_train_split(x, y, random_state=42):
    """Create a train/test split using the realistic cluster-based method."""
    data = pd.DataFrame({"sequence": x, "binding_scores": y})
    split_data = create_split(data, random_state=random_state)
    return split_data

def fit_surrogate_model(model, train_df: pd.DataFrame, token_to_idx: Dict[str, int]):
    x_train = np.stack(
        [encode_sequence(sequence, token_to_idx) for sequence in train_df["sequence"]]
    )
    x_train_one_hot = np.stack([one_hot_encode_sequence(seq, num_tokens=len(token_to_idx)) for seq in x_train])
    y_train = train_df["binding_scores"].to_numpy(dtype=np.float32)
    model.fit(x_train_one_hot, y_train)
    return model

def predict_surrogate(model, sequence: str, token_to_idx: Dict[str, int]) -> float:
    encoded = encode_sequence(sequence, token_to_idx).reshape(1, -1)
    one_hot_encoded = one_hot_encode_sequence(encoded.flatten(), num_tokens=len(token_to_idx)).reshape(1, -1)
    prediction = model.predict(one_hot_encoded)
    return float(np.asarray(prediction).reshape(-1)[0])

def main():
    # Load dataset
    alphabet = ["A", "C", "G", "T"]
    token_to_idx = {token: idx for idx, token in enumerate(alphabet)}

    x, y = load_data("tfbind8")

    df = build_tfbind8_dataframe(x, y, alphabet)
    train_df = df[df["split"] == "train"].copy()

    print('Dataset Loaded and Oracle Initialized. Starting model training and evaluation...')

    _random_forest = random_forest.RandomForestModel(n_estimators=500)
    _mlp = mlp.MLPModel()
    # Train models
    rf_model = fit_surrogate_model(_random_forest, train_df, token_to_idx)
    mlp_model = fit_surrogate_model(_mlp, train_df, token_to_idx)

    print('Models trained. Starting evaluation on test set...')

    # Evaluate models on test set
    test_data = df[df["split"] != "train"].copy()

    # Initialize new columns
    test_data['min_hamming_distance'] = 0
    test_data['rf_prediction'] = 0.0
    test_data['mlp_prediction'] = 0.0

    # Calculate minimum Hamming distance to training set for each test sequence
    test_data['min_hamming_distance'] = min_hamming_distance(
        np.stack([encode_sequence(seq, token_to_idx) for seq in test_data['sequence']]),
        np.stack([encode_sequence(seq, token_to_idx) for seq in train_df['sequence']])
    )

    for seq in tqdm(test_data['sequence']):
        score_rf = predict_surrogate(rf_model, seq, token_to_idx)
        score_mlp = predict_surrogate(mlp_model, seq, token_to_idx)
        test_data.loc[test_data['sequence'] == seq, 'rf_prediction'] = score_rf
        test_data.loc[test_data['sequence'] == seq, 'mlp_prediction'] = score_mlp

 
    test_data.to_csv("results/tfbind8_sequences_scored.csv", index=False)

if __name__ == "__main__":
    main()
