# ml_models.py
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import TimeDistributed, Bidirectional
from tensorflow.keras.optimizers import Adam

# ---------------------------
# Helper: reshape sequences
# ---------------------------
def reshape_for_rnn(X, timesteps=10, features=None):
    """
    Reshape data for RNN-based models.
    Input:  (samples, features)
    Output: (samples, timesteps, features)
    """
    if features is None:
        features = X.shape[1]

    n_samples = X.shape[0] - timesteps
    X_seq = np.zeros((n_samples, timesteps, features))
    for i in range(n_samples):
        X_seq[i] = X[i:i+timesteps]
    return X_seq


# ---------------------------
# 1. LSTM
# ---------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(64, activation="tanh", return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, activation="tanh"),
        Dense(2)  # predicting [latitude, longitude]
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# ---------------------------
# 2. CNN + RNN (CNN + LSTM)
# ---------------------------
def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        LSTM(64, activation="tanh"),
        Dense(32, activation="relu"),
        Dense(2)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# ---------------------------
# 3. LSTM + GRU
# ---------------------------
def build_lstm_gru(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        GRU(32),
        Dense(2)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# ---------------------------
# 4. TCN (Temporal Convolutional Network)
# ---------------------------
def build_tcn(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding="causal", activation="relu", input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding="causal", activation="relu"),
        Conv1D(filters=64, kernel_size=3, dilation_rate=4, padding="causal", activation="relu"),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(2)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# ---------------------------
# 5. Bidirectional LSTM
# ---------------------------
def build_bilstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, activation="tanh", return_sequences=True), input_shape=input_shape),
        Dropout(0.2),
        Bidirectional(LSTM(32, activation="tanh")),
        Dense(32, activation="relu"),
        Dense(2)  # predicting [latitude, longitude]
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# ---------------------------
# Wrapper for easy model selection
# ---------------------------
def get_model(model_name, input_shape=None):
    """
    Returns the model instance based on name.
    For ML models → input_shape is required.
    """
    model_name = model_name.lower()

    if model_name == "lstm":
        return build_lstm(input_shape)
    elif model_name in ["cnn+rnn", "cnn_lstm"]:
        return build_cnn_lstm(input_shape)
    elif model_name == "lstm+gru":
        return build_lstm_gru(input_shape)
    elif model_name == "tcn":
        return build_tcn(input_shape)
    elif model_name in ["bilstm", "bi-lstm"]:
        return build_bilstm(input_shape)
    else:
        raise ValueError(f"Unknown model: {model_name}")


# ---------------------------
# Quick test
# ---------------------------
if __name__ == "__main__":
    input_shape = (10, 9)  # 10 timesteps, 9 features (ax, ay, az, wx, wy, wz, Bx, By, Bz)

    lstm_model = get_model("lstm", input_shape)
    print("LSTM summary:")
    lstm_model.summary()

    tcn_model = get_model("tcn", input_shape)
    print("\nTCN summary:")
    tcn_model.summary()

    bilstm_model = get_model("bilstm", input_shape)
    print("\nBi-LSTM summary:")
    bilstm_model.summary()