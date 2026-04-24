import torch
import torch.nn as nn
import torch.nn.functional as F

class GoalDirectedLSTM(nn.Module):
    def __init__(self, vocab_size, sequence_length, embedding_dim, hidden_dim, goal_score):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.vocab_size = vocab_size
        self.lstm = nn.LSTM(embedding_dim + goal_score, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, seq, goal_score, hidden=None):
        embeddings = self.embedding(seq)
        goal_embeddings = goal_score.unsqueeze(1).expand(-1, self.sequence_length, -1)
        input = torch.cat((embeddings, goal_embeddings), dim=-1)
        output, hidden = self.lstm(input, hidden)
        logits = self.fc(output)
        return logits, hidden

    def generate(self, goal_score, temperature = 1.0):
        self.eval()
        input_seq = torch.zeros((1, 1), dtype=torch.long)
        goal_tensor = torch.tensor([[goal_score]], dtype=torch.float)
        generated_sequence = []
        hidden = None

        for _ in range(self.sequence_length):
            logits, hidden = self.forward(input_seq, goal_tensor, hidden)
            next_token_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated_sequence.append(next_token.item())
            input_seq = next_token
        return generated_sequence
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
