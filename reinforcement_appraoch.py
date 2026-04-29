import numpy as np
import pandas as pd
import tensorflow as tf

import torch
from torch.utils.data import TensorDataset, DataLoader
from models import oracle, random_forest, goal_directed_lstm
from assets.data_ops import decode_sequence, encode_sequence, one_hot_encode_sequence, one_hot_decode_sequence, load_data, build_tfbind8_dataframe


def train_goal_directed_model_v2(model, loader, epochs=20, lr=0.001, entropy_weight=0.01):
    """Training with entropy regularization to prevent mode collapse."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()

            input = batch_x[:, :-1]
            target_part = batch_x[:, 1:]
            logits, _ = model(input, batch_y)
            ce_loss = criterion(logits.reshape(-1, model.vocab_size), target_part.reshape(-1))

            # Entropy regularization: encourage diverse token predictions
            # penalize low-entropy output distris during training to prevent mode collapse
            probs = torch.softmax(logits, dim=-1)
            entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
            loss = ce_loss - entropy_weight * entropy  # maximize entropy

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        if epoch % 5 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(loader):.4f}')

def diversity_filter(sequences, scores, real_scores, min_hamming=2):
    """Greedily pick sequences that are at least min_hamming apart."""
    selected_idx = [0]
    for i in range(1, len(sequences)):
        is_diverse = True
        for j in selected_idx:
            dist = np.sum(sequences[i] != sequences[j])
            if dist < min_hamming:
                is_diverse = False
                break
        if is_diverse:
            selected_idx.append(i)
    idx = np.array(selected_idx)
    return sequences[idx], scores[idx], real_scores[idx]


def rl_loop_v2(model, scorer, oracle, num_iterations=6, batch_size=128,
               X_train=None, y_train=None, GB1=False,
               min_hamming=2, replay_fraction=0.3, temp_start=0.8, temp_end=1.5,
               entropy_weight=0.01):
    """Improved RL loop"""
    values = {'mutation': [], 'predicted_score': [], 'real_score': [], 'iteration': []}
    
    # Keep original training data for replay
    replay_x = torch.LongTensor(X_train)
    replay_y = torch.FloatTensor(y_train).view(-1, 1)
    
    for i in range(num_iterations):
        # Anneal temperature: increase over iterations to maintain exploration
        temp = temp_start + (temp_end - temp_start) * (i / max(num_iterations - 1, 1))
        print(f'-------- Iteration {i+1}/{num_iterations} (temp={temp:.2f}) --------')
        
        ### Propose new sequences
        new_sequences = []
        for _ in range(batch_size):
            seq = model.generate(goal_score=1.1, temperature=temp)
            new_sequences.append(seq)

        new_sequences_np = np.array(new_sequences)
        
        ### Score new sequences by RF
        scores = np.array([scorer.predict(seq.reshape(1, -1)) for seq in new_sequences_np])

        ### Lookup oracle for real scores
        real_scores = []
        for seq in new_sequences_np:
            ##TODO: oracle lookup muss geändert werden
            if GB1:
                seq_one_hot = tf.one_hot(seq, depth=20).numpy()
                real_score = oracle.inference(torch.tensor(seq_one_hot.flatten(), dtype=torch.float32)).item()
            else:
                seq_str = decode_sequence(seq, alphabet=["A", "C", "G", "T"])
                real_score = oracle.evaluate(seq_str)
            real_scores.append(real_score)
        real_scores_np = np.array(real_scores)

        ### Select top 20% sequences based on (real) scores
        # Edit: use RF scores for selection (real_scores_np -> scores)
        threshold = np.percentile(scores, 80)
        top_mask = scores >= threshold
        top_mask = np.asarray(top_mask).astype(bool).ravel()
        top_sequences = new_sequences_np[top_mask, :]
        top_scores = scores[top_mask]
        top_real_scores = real_scores_np[top_mask]

        ### Diversity filter: remove near-duplicates
        if len(top_sequences) > 1:
            top_sequences, top_scores, top_real_scores = diversity_filter(
                top_sequences, top_scores, top_real_scores, min_hamming=min_hamming
            )

        n_unique = len(top_sequences)
        print(f'Avg proposed: {np.mean(scores):.4f}, Avg real: {np.mean(real_scores_np):.4f}, '
              f'Top 20% avg: {np.mean(top_scores):.4f}, Unique diverse: {n_unique}')

        # Append for plotting
        values['mutation'].extend(top_sequences)
        values['predicted_score'].extend(top_scores)
        values['real_score'].extend(top_real_scores)
        values['iteration'].extend([i] * n_unique)
        
        #mix new top sequences with original training data
        x_new = torch.LongTensor(top_sequences)
        y_new = torch.FloatTensor(top_scores).view(-1, 1)
        
        n_replay = int(len(x_new) * replay_fraction / (1 - replay_fraction))
        n_replay = min(n_replay, len(replay_x))
        replay_idx = torch.randperm(len(replay_x))[:n_replay]
        
        x_combined = torch.cat([x_new, replay_x[replay_idx]])
        y_combined = torch.cat([y_new, replay_y[replay_idx]])
        
        dataloader = DataLoader(TensorDataset(x_combined, y_combined), batch_size=16, shuffle=True)
        train_goal_directed_model_v2(model, dataloader, epochs=20, lr=0.001, entropy_weight=entropy_weight)

    return values


def run_rl_experiment_lstm_rf():
    '''Run the RL loop with LSTM generator and RF surrogate.
        1. Load data
        2. Train RF surrogate on trainig data
        3. Pretrain LSTM on trainig data
        4. Run RL loop with oracle feedback and RF surrogate for selection
        
        Retruns:
            values: dict with generated sequences, predicted scores, real scores, and iteration numbers for plotting
    '''
    # Load data and prepare DataFrame
    x, y = load_data(name="tfbind8")
    df = build_tfbind8_dataframe(x, y, alphabet=["A", "C", "G", "T"])
    train_df = df[df["split"] == "train"].copy()
    token_to_idx = {token: idx for idx, token in enumerate(["A", "C", "G", "T"])}
    X_train = np.array([encode_sequence(seq, token_to_idx) for seq in train_df["sequence"]])
    y_train = train_df["binding_scores"].values
    
    # Train RF surrogate
    surrogate = random_forest.RandomForestModel(n_estimators=200, random_state=42)
    surrogate.fit(X_train, y_train)

    # Pretrain LSTM generator
    lstm_model = goal_directed_lstm.GoalDirectedLSTM(vocab_size=4, embedding_dim=16, sequence_length=len(X_train[0]), hidden_dim=32, goal_score=1)

    X_train_tensor = torch.LongTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    train_goal_directed_model_v2(lstm_model, train_loader, epochs=100)

    # Create oracle
    oracle_TF_bind = oracle.Oracle_TFBind8(df)

    # Run RL loop
    values = rl_loop_v2(
        model=lstm_model,
        scorer=surrogate,
        oracle=oracle_TF_bind,
        num_iterations=10,
        batch_size=265,
        X_train=X_train,
        y_train=y_train,
        GB1=False,
        min_hamming=2,
        replay_fraction=0.1,
        temp_start=0.5,
        temp_end=1.2,
        entropy_weight=0.1
    )
    return values
