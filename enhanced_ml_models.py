# enhanced_ml_models.py
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.layers import Bidirectional, TimeDistributed, Input, Concatenate, Add
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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
# Callbacks for training deep learning models
# ---------------------------
def get_callbacks():
    """
    Returns a set of callbacks for training deep learning models:
    - Early stopping to prevent overfitting
    - Learning rate reduction on plateau
    """
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    lr_reducer = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, lr_reducer]

# ---------------------------
# 1. Enhanced LSTM with Bidirectional layers
# ---------------------------
def build_bidirectional_lstm(input_shape):
    model = Sequential([
        Bidirectional(LSTM(64, activation="tanh", return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(32, activation="tanh")),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(2)  # predicting [latitude, longitude]
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# ---------------------------
# 2. CNN + LSTM (Improved)
# ---------------------------
def build_cnn_lstm(input_shape):
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu", padding='same', input_shape=input_shape),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=128, kernel_size=3, activation="relu", padding='same'),
        MaxPooling1D(pool_size=2),
        LSTM(128, activation="tanh", return_sequences=True),
        Dropout(0.3),
        LSTM(64, activation="tanh"),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model

# ---------------------------
# 3. LSTM + GRU (Stacked)
# ---------------------------
def build_lstm_gru(input_shape):
    model = Sequential([
        LSTM(128, return_sequences=True, input_shape=input_shape),
        Dropout(0.3),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        GRU(32),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(2)
    ])
    model.compile(optimizer=Adam(5e-4), loss="mse")
    return model

# ---------------------------
# 4. TCN (Temporal Convolutional Network)
# ---------------------------
def build_tcn(input_shape):
    # Improved TCN using stacked dilated CNN layers
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, dilation_rate=1, padding='causal', 
               activation="relu", input_shape=input_shape),
        Conv1D(filters=64, kernel_size=3, dilation_rate=2, padding='causal', activation="relu"),
        Conv1D(filters=64, kernel_size=3, dilation_rate=4, padding='causal', activation="relu"),
        Conv1D(filters=64, kernel_size=3, dilation_rate=8, padding='causal', activation="relu"),
        Dropout(0.2),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dense(2)
    ])
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

# ---------------------------
# 5. Transformer Model for Time Series
# ---------------------------
def build_transformer(input_shape):
    """
    Build a Transformer model for time series forecasting.
    """
    inputs = Input(shape=input_shape)
    
    # Positional encoding could be added here
    x = inputs
    
    # Transformer blocks
    for _ in range(2):  # 2 encoder blocks
        # Multi-head attention
        attention_output = MultiHeadAttention(
            num_heads=4, key_dim=32
        )(x, x)
        x = Add()([x, attention_output])
        x = LayerNormalization(epsilon=1e-6)(x)
        
        # Feed-forward network
        ffn = Dense(128, activation="relu")(x)
        ffn = Dense(input_shape[-1])(ffn)
        x = Add()([x, ffn])
        x = LayerNormalization(epsilon=1e-6)(x)
    
    # Output layers
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model

# ---------------------------
# 6. Hybrid CNN-Transformer Model
# ---------------------------
def build_cnn_transformer(input_shape):
    """
    A hybrid model combining CNN feature extraction with Transformer attention.
    """
    inputs = Input(shape=input_shape)
    
    # CNN feature extraction
    x = Conv1D(filters=64, kernel_size=3, activation="relu", padding="same")(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    
    # Transformer blocks
    attention_output = MultiHeadAttention(num_heads=4, key_dim=32)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Feed-forward network
    ffn = Dense(128, activation="relu")(x)
    ffn = Dense(128)(ffn)
    x = Add()([x, ffn])
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # Output layers
    x = Flatten()(x)
    x = Dense(64, activation="relu")(x)
    x = Dropout(0.2)(x)
    outputs = Dense(2)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model

# ---------------------------
# 7. ResNet-style 1D CNN
# ---------------------------
def build_resnet1d(input_shape):
    """
    ResNet-style model with 1D convolutions for time series.
    """
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(64, kernel_size=7, padding='same')(inputs)
    x = LayerNormalization()(x)
    
    # Residual blocks
    for filters in [64, 128, 256]:
        # Residual block 1
        residual = x
        x = Conv1D(filters, kernel_size=3, padding='same', activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Conv1D(filters, kernel_size=3, padding='same')(x)
        
        # If shapes don't match, transform residual
        if residual.shape[-1] != filters:
            residual = Conv1D(filters, kernel_size=1, padding='same')(residual)
            
        x = Add()([x, residual])
        x = LayerNormalization()(x)
    
    # Output layers
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(2)(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(1e-4), loss="mse")
    return model

# ---------------------------
# 8. Ensemble of Deep Learning Models
# ---------------------------
class DeepEnsembleModel:
    """
    Ensemble of multiple deep learning models.
    """
    def __init__(self, models):
        """
        Initialize with a list of compiled Keras models.
        """
        self.models = models
    
    def fit(self, X, y, epochs=50, batch_size=32, verbose=1, validation_split=0.1):
        """
        Train each model in the ensemble.
        """
        callbacks = get_callbacks()
        histories = []
        
        for i, model in enumerate(self.models):
            print(f"\nTraining model {i+1}/{len(self.models)}...")
            history = model.fit(
                X, y, 
                epochs=epochs,
                batch_size=batch_size,
                verbose=verbose,
                validation_split=validation_split,
                callbacks=callbacks
            )
            histories.append(history)
        
        return histories
    
    def predict(self, X):
        """
        Average predictions from all models.
        """
        preds = [model.predict(X) for model in self.models]
        return np.mean(preds, axis=0)

# ---------------------------
# 9. XGBoost with MultiOutput Wrapper
# ---------------------------
def build_xgboost():
    base_model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    # Wrap the model for multi-output regression
    return MultiOutputRegressor(base_model)

# ---------------------------
# 10. Gradient Boosting Regressor
# ---------------------------
def build_gradient_boosting():
    base_model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42
    )
    return MultiOutputRegressor(base_model)

# ---------------------------
# 11. KNN Regressor
# ---------------------------
def build_knn():
    return KNeighborsRegressor(
        n_neighbors=5,
        weights='distance',
        algorithm='auto',
        p=2  # Euclidean distance
    )

# ---------------------------
# Wrapper for easy model selection
# ---------------------------
def get_model(model_name, input_shape=None):
    """
    Returns the model instance based on name.
    Args:
        model_name (str): Name of the model
        input_shape (tuple): Shape of the input data (required for deep learning models)
    
    Returns:
        Model instance
    """
    model_name = model_name.lower()

    if model_name == "lstm":
        return build_bidirectional_lstm(input_shape)
    elif model_name in ["cnn+rnn", "cnn_lstm"]:
        return build_cnn_lstm(input_shape)
    elif model_name == "lstm+gru":
        return build_lstm_gru(input_shape)
    elif model_name == "tcn":
        return build_tcn(input_shape)
    elif model_name == "transformer":
        return build_transformer(input_shape)
    elif model_name == "cnn+transformer":
        return build_cnn_transformer(input_shape)
    elif model_name == "resnet1d":
        return build_resnet1d(input_shape)
    elif model_name == "ensemble":
        # Create an ensemble of 3 different architectures
        models = [
            build_bidirectional_lstm(input_shape),
            build_cnn_lstm(input_shape),
            build_transformer(input_shape)
        ]
        return DeepEnsembleModel(models)
    elif model_name == "xgboost":
        return build_xgboost()
    elif model_name == "gradient_boosting":
        return build_gradient_boosting()
    elif model_name == "knn":
        return build_knn()
    elif model_name == "svr":
        return SVR(kernel="rbf", C=100, gamma=0.1, epsilon=0.1)
    elif model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42)
    else:
        raise ValueError(f"Unknown model: {model_name}")
