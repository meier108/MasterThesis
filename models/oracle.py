import pandas as pd


class Oracle_TFBind8:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self._scores = dict(zip(data["sequence"], data["binding_scores"]))
        self._splits = dict(zip(data["sequence"], data["split"]))

    def exists(self, sequence: str) -> bool:
        """Check if the given sequence exists in the dataset."""
        return sequence in self._scores

    def get_score(self, sequence: str):
        """Return the binding score for the given sequence, or None if missing."""
        return self._scores.get(sequence)

    def evaluate(self, sequence: str):
        """Compatibility wrapper used by the optimization loop."""
        return self.get_score(sequence)

    def get_split(self, sequence: str):
        """Return the split label for a sequence, or None if unknown."""
        return self._splits.get(sequence)
    
    def get_df(self):
        """Return the underlying DataFrame."""
        return self.data


