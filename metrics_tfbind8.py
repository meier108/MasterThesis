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
from data.create_split import create_split



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
    test_data = df[df["split"] != "train"]
    rf_predictions = [predict_surrogate(rf_model, seq, token_to_idx) for seq in test_data['sequence']]
    mlp_predictions = [predict_surrogate(mlp_model, seq, token_to_idx) for seq in test_data['sequence']]
    oracle_scores = test_data['binding_scores'].to_numpy(dtype=np.float32)

    #save predictions and oracle scores to csv
    predictions_df = pd.DataFrame({
        "sequence": test_data['sequence'],
        "oracle_score": oracle_scores,
        "rf_prediction": rf_predictions,
        "mlp_prediction": mlp_predictions,
        "split": test_data['split']
    })

    predictions_df.to_csv("results/predictions_TFBind8_v2.csv", index=False)

    # Calculate metrics -> for complete Test Data
    rf_spearmanr = calculate_spearmanr(oracle_scores, rf_predictions)
    mlp_spearmanr = calculate_spearmanr(oracle_scores, mlp_predictions)
    
    rf_mse = calculate_mse(oracle_scores, rf_predictions)
    mlp_mse = calculate_mse(oracle_scores, mlp_predictions)
    
    rf_bias = calculate_bias(oracle_scores, rf_predictions)
    mlp_bias = calculate_bias(oracle_scores, mlp_predictions)

    # Print results
    print(f"Random Forest - Spearman's r: {rf_spearmanr:.4f}, MSE: {rf_mse:.4f}, Bias: {rf_bias:.4f}")
    print(f"MLP - Spearman's r: {mlp_spearmanr:.4f}, MSE: {mlp_mse:.4f}, Bias: {mlp_bias:.4f}")
    
    # Store results in metrics.csv
    results_df = pd.DataFrame({
        "model": ["Random Forest", "MLP"],
        "spearmanr": [rf_spearmanr, mlp_spearmanr],
        "mse": [rf_mse, mlp_mse],
        "bias": [rf_bias, mlp_bias]
    })

    # Calculate metrics for each split
    for split in test_data['split'].unique():
        split_data = test_data[test_data['split'] == split]
        split_oracle_scores = split_data['binding_scores'].to_numpy(dtype=np.float32)
        split_rf_predictions = [predict_surrogate(rf_model, seq, token_to_idx) for seq in split_data['sequence']]
        split_mlp_predictions = [predict_surrogate(mlp_model, seq, token_to_idx) for seq in split_data['sequence']]

        split_rf_spearmanr = calculate_spearmanr(split_oracle_scores, split_rf_predictions)
        split_mlp_spearmanr = calculate_spearmanr(split_oracle_scores, split_mlp_predictions)
        
        split_rf_mse = calculate_mse(split_oracle_scores, split_rf_predictions)
        split_mlp_mse = calculate_mse(split_oracle_scores, split_mlp_predictions)
        
        split_rf_bias = calculate_bias(split_oracle_scores, split_rf_predictions)
        split_mlp_bias = calculate_bias(split_oracle_scores, split_mlp_predictions)

        print(f"Split: {split} - Random Forest - Spearman's r: {split_rf_spearmanr:.4f}, MSE: {split_rf_mse:.4f}, Bias: {split_rf_bias:.4f}")
        print(f"Split: {split} - MLP - Spearman's r: {split_mlp_spearmanr:.4f}, MSE: {split_mlp_mse:.4f}, Bias: {split_mlp_bias:.4f}")

        # Append split results to metrics.csv
        split_results_df = pd.DataFrame({
            "model": [f"Random Forest ({split})", f"MLP ({split})"],
            "spearmanr": [split_rf_spearmanr, split_mlp_spearmanr],
            "mse": [split_rf_mse, split_mlp_mse],
            "bias": [split_rf_bias, split_mlp_bias]
        })
        results_df = pd.concat([results_df, split_results_df], ignore_index=True)

    results_df.to_csv("results/metrics_TFBind8_v2.csv", index=False)

if __name__ == "__main__":
    main()
