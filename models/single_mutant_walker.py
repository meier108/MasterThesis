import numpy as np

class SingleMutantWalker:
    """A simple random walk model that generates single mutants of a given sequence. 
    It randomly mutates one position in the input sequence to create a new sequence.
    """

    def __init__(self, alphabet, sequence_length):
        """Initialize the SingleMutantWalker with the given alphabet and sequence length."""
        self.alphabet = alphabet
        self.sequence_length = sequence_length

    def mutate_sequence(self, sequence):
        sequence = str(sequence).strip().upper()
        if len(sequence) != self.sequence_length:
            raise ValueError(
                f"Expected sequence length {self.sequence_length}, got {len(sequence)}"
            )

        # Randomly select a position to mutate
        pos = np.random.randint(0, self.sequence_length)

        # Select a character different from current position.
        candidates = [c for c in self.alphabet if c != sequence[pos]]
        new_char = np.random.choice(candidates if candidates else self.alphabet)

        # Create a new sequence with the mutation.
        mutated_sequence = list(sequence)
        mutated_sequence[pos] = new_char
        return "".join(mutated_sequence)
