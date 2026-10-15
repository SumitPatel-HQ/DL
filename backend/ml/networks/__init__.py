"""CPU LSTM classifiers. No BiLSTM, CNN or CTC in P0."""

from ml.networks.lstm import LstmClassifier, parameter_count

__all__ = ["LstmClassifier", "parameter_count"]
