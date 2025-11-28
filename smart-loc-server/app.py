"""
Flask server for Smart Location Predictor (GRU encoder + XGBoost regressor).
- Receives streaming IMU samples from phone via POST /stream
- Optionally receives GPS with samples (used as labels)
- Background trainer retrains periodically when enough labeled windows exist
- POST /predict returns predicted latitude & longitude for the most recent window
- POST /gps echoes phone GPS if provided, otherwise returns "No signal"

Run:
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
$ python app.py

From Android emulator use http://10.0.2.2:5001 for requests.
From real phone use http://<your_pc_ip>:5000
"""

import os
import time
import threading
import json
import pickle
from collections import deque
from io import StringIO

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import dump, load
from sklearn.preprocessing import StandardScaler
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

# -----------------------
# Configurable constants
# -----------------------
WINDOW_LENGTH = 10               # number of samples per sequence window
SENSOR_DIM = 9                   # ax,ay,az,gx,gy,gz,mx,my,mz
BUFFER_MAX_SAMPLES = 2000        # keep last N raw samples in rolling buffer
MIN_LABELED_WINDOWS = 8         # minimum labeled windows before training
TRAIN_INTERVAL = 20.0            # seconds between background trainer runs
EPOCHS_GRU = 8                    # epochs for GRU autoencoder training each round
BATCH_SIZE = 32
EMBED_DIM = 64
XGB_PARAMS = {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.05}
MODEL_SAVE_PATH = "models/gru_xgb_model.pkl"
SCALER_SAVE_PATH = "models/scaler.pkl"

# -----------------------
# Server state / buffers
# -----------------------
# Rolling buffer of raw samples (each sample is a dict or np.array of length SENSOR_DIM)
raw_buffer = deque(maxlen=BUFFER_MAX_SAMPLES)   # contains tuples (timestamp, sensor_array)
raw_buffer_lock = threading.Lock()

# Labeled windows accumulated for supervised training:
# each entry is (window_array shape=(WINDOW_LENGTH,SENSOR_DIM), lat, lon)
labeled_windows = []
labeled_lock = threading.Lock()

# Latest GPS posted (most recent GPS received with any /stream or /gps call)
latest_gps = None
latest_gps_lock = threading.Lock()

# Model holders
gru_encoder_model = None   # Keras model that maps (WINDOW_LENGTH,SENSOR_DIM) -> embedding (EMBED_DIM,)
gru_autoencoder = None     # full autoencoder model (for training)
xgb_model = None           # MultiOutputRegressor wrapping xgboost.XGBRegressor
scaler = None              # StandardScaler for feature normalization (fitted on-window features)
model_lock = threading.Lock()

# Utility flag
stop_background = False

# -----------------------
# Flask app
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# Helpers: GRU autoencoder
# -----------------------
def build_gru_autoencoder(window_len=WINDOW_LENGTH, sensor_dim=SENSOR_DIM, embed_dim=EMBED_DIM):
    """
    Build a simple GRU autoencoder:
      Input -> GRU encoder -> embedding -> RepeatVector -> GRU decoder -> TimeDistributed(Dense(sensor_dim))
    Returns: (autoencoder_model, encoder_model)
    """
    inputs = layers.Input(shape=(window_len, sensor_dim), name="input_sequence")
    # Encoder
    x = layers.GRU(128, return_sequences=True, name="enc_gru1")(inputs)
    x = layers.GRU(64, return_sequences=False, name="enc_gru2")(x)
    embedding = layers.Dense(embed_dim, activation="tanh", name="embedding")(x)

    # Decoder
    repeated = layers.RepeatVector(window_len, name="repeat")(embedding)
    y = layers.GRU(64, return_sequences=True, name="dec_gru1")(repeated)
    y = layers.GRU(128, return_sequences=True, name="dec_gru2")(y)
    outputs = layers.TimeDistributed(layers.Dense(sensor_dim), name="reconstruction")(y)

    autoencoder = models.Model(inputs=inputs, outputs=outputs, name="gru_autoencoder")
    encoder = models.Model(inputs=inputs, outputs=embedding, name="gru_encoder")

    autoencoder.compile(optimizer=optimizers.Adam(1e-3), loss=losses.MeanSquaredError())
    return autoencoder, encoder

# -----------------------
# Helpers: saving/loading models
# -----------------------
def save_models(encoder, xgb_model_obj, scaler_obj, path=MODEL_SAVE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        "xgb": xgb_model_obj,
    }
    # Save encoder weights separately (Keras model)
    encoder.save("models/gru_encoder_keras.keras", include_optimizer=False)
    # Save other pieces
    with open(path, "wb") as f:
        pickle.dump(save_dict, f)
    dump(scaler_obj, SCALER_SAVE_PATH)
    app.logger.info("Saved models to disk.")

def load_models(path=MODEL_SAVE_PATH):
    global gru_encoder_model, xgb_model, scaler
    if os.path.exists("models/gru_encoder_keras.keras"):
        try:
            gru_encoder_model = tf.keras.models.load_model("models/gru_encoder_keras.keras")
            app.logger.info("Loaded GRU encoder Keras model from models/gru_encoder_keras")
        except Exception as e:
            app.logger.warning(f"Failed to load keras encoder: {e}")
            gru_encoder_model = None
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                d = pickle.load(f)
                xgb = d.get("xgb", None)
                if xgb is not None:
                    app.logger.info("Loaded XGBoost model from pickle.")
                    return xgb
        except Exception as e:
            app.logger.warning(f"Failed to load xgb pickle: {e}")
    if os.path.exists(SCALER_SAVE_PATH):
        try:
            scaler_loaded = load(SCALER_SAVE_PATH)
            app.logger.info("Loaded scaler.")
            scaler = scaler_loaded
        except Exception as e:
            app.logger.warning(f"Failed to load scaler: {e}")
    return None

# -----------------------
# Preprocessing helpers
# -----------------------
def sample_to_array(sample_dict):
    """
    sample_dict must contain the 9 keys in order:
    ax, ay, az, gx, gy, gz, mx, my, mz
    Returns numpy array shape (SENSOR_DIM,)
    """
    arr = np.array([
        float(sample_dict.get("ax", 0.0)),
        float(sample_dict.get("ay", 0.0)),
        float(sample_dict.get("az", 0.0)),
        float(sample_dict.get("gx", 0.0)),
        float(sample_dict.get("gy", 0.0)),
        float(sample_dict.get("gz", 0.0)),
        float(sample_dict.get("mx", 0.0)),
        float(sample_dict.get("my", 0.0)),
        float(sample_dict.get("mz", 0.0)),
    ], dtype=np.float32)
    return arr

def windows_from_buffer(window_length=WINDOW_LENGTH):
    """
    Create contiguous (non-overlapping) windows from raw_buffer.
    Returns list of np arrays shape (window_length, SENSOR_DIM)
    """
    with raw_buffer_lock:
        raw_list = list(raw_buffer)
    # raw_list entries are (timestamp, sensor_array)
    arrays = [x[1] for x in raw_list]
    if len(arrays) < window_length:
        return []
    windows = []
    # sliding windows with step=1 (overlapping) - helps training
    for start in range(0, len(arrays) - window_length + 1):
        w = np.stack(arrays[start:start + window_length], axis=0)
        windows.append(w)
    return windows

# -----------------------
# Background trainer thread
# -----------------------
def background_trainer():
    global gru_autoencoder, gru_encoder_model, xgb_model, scaler
    app.logger.info("Background trainer thread started.")
    while not stop_background:
        try:
            # Check if we have enough labeled windows
            with labeled_lock:
                n_labeled = len(labeled_windows)
            app.logger.debug(f"Trainer awakened: {n_labeled} labeled windows available.")
            if n_labeled >= MIN_LABELED_WINDOWS:
                # Prepare X (windows) and Y (lat,long)
                with labeled_lock:
                    data_copy = labeled_windows.copy()
                    labeled_windows.clear()  # consume them (we keep them consumed; you could keep history instead)

                X = np.stack([entry[0] for entry in data_copy], axis=0)  # shape (M, window_len, sensor_dim)
                Y = np.array([[entry[1], entry[2]] for entry in data_copy], dtype=np.float32)  # (M,2)

                # Flatten windows for scaler fitting: each window -> (window_len * sensor_dim,)
                M = X.shape[0]
                X_flat = X.reshape(M, -1)
                scaler = StandardScaler()
                X_flat_scaled = scaler.fit_transform(X_flat)
                # reshape back if needed (encoder expects per-window 2D time series)
                # We'll normalize per-feature across flattened windows, but encoder expects time series values.
                X_scaled = X_flat_scaled.reshape(M, X.shape[1], X.shape[2])

                # Build or re-init GRU autoencoder + encoder
                gru_autoencoder, gru_encoder_model = build_gru_autoencoder(window_len=X.shape[1], sensor_dim=X.shape[2], embed_dim=EMBED_DIM)
                # Train autoencoder to reconstruct windows
                app.logger.info(f"Training GRU autoencoder on {M} windows for {EPOCHS_GRU} epochs...")
                gru_autoencoder.fit(X_scaled, X_scaled, epochs=EPOCHS_GRU, batch_size=BATCH_SIZE, verbose=1)

                # Generate embeddings with encoder
                embeddings = gru_encoder_model.predict(X_scaled, batch_size=128)
                # Train XGBoost regressor on embeddings -> lat/lon
                app.logger.info("Training XGBoost regressor on encoder embeddings...")
                xgb_reg = xgb.XGBRegressor(objective="reg:squarederror", **XGB_PARAMS, verbosity=0)
                xgb_multi = MultiOutputRegressor(xgb_reg)
                xgb_multi.fit(embeddings, Y)

                # Save models (encoder weights saved via Keras, xgb + scaler pickled)
                with model_lock:
                    xgb_model = xgb_multi
                    # persist models to disk
                    os.makedirs("models", exist_ok=True)
                    save_models(gru_encoder_model, xgb_model, scaler, MODEL_SAVE_PATH)
                app.logger.info("Background training completed and models saved.")
            else:
                app.logger.debug("Not enough labeled windows yet for training.")
        except Exception as e:
            app.logger.exception("Exception in background trainer loop:")
        # Sleep until next check
        time.sleep(TRAIN_INTERVAL)
    app.logger.info("Background trainer thread stopping.")

# -----------------------
# Endpoint: stream IMU samples and optional GPS labels
# -----------------------
@app.route("/stream", methods=["POST"])
def stream_sample():
    """
    Expects JSON payload with structure:
    {
      "sample": {"ax":..., "ay":..., "az":..., "gx":..., "gy":..., "gz":..., "mx":..., "my":..., "mz":...},
      "timestamp": 169... (optional),
      "gps": {"lat": ..., "lon": ...}   (optional)
    }

    The phone should POST samples at model sampling rate (e.g., 10 Hz).
    If 'gps' is present, server will attach the latest window -> label pair to labeled_windows for training.
    """
    global latest_gps
    data = None
    try:
        data = request.get_json(force=True)
    except Exception:
        return jsonify({"status": "error", "message": "invalid json"}), 400

    if not data or "sample" not in data:
        return jsonify({"status": "error", "message": "no sample provided"}), 400

    sample = data["sample"]
    timestamp = data.get("timestamp", time.time())
    gps = data.get("gps", None)

    # convert to array
    arr = sample_to_array(sample)
    with raw_buffer_lock:
        raw_buffer.append((timestamp, arr))

    if gps:
        try:
            lat = float(gps.get("lat"))
            lon = float(gps.get("lon"))
            with latest_gps_lock:
                latest_gps = {"lat": lat, "lon": lon, "ts": timestamp}
            # Try to create a window ending at the latest sample
            with raw_buffer_lock:
                raw_list = list(raw_buffer)
            if len(raw_list) >= WINDOW_LENGTH:
                # take last WINDOW_LENGTH samples as labeled window
                window_arr = np.stack([x[1] for x in raw_list[-WINDOW_LENGTH:]], axis=0)
                with labeled_lock:
                    labeled_windows.append((window_arr, lat, lon))
                app.logger.debug("Added labeled window for training.")
        except Exception as e:
            app.logger.warning("Invalid gps data provided in stream.")

    return jsonify({"status": "ok", "buffer_len": len(raw_buffer), "labeled_windows": len(labeled_windows)})

# -----------------------
# Endpoint: predict (ML Model)
# -----------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Server uses the most recent WINDOW_LENGTH samples in raw_buffer to predict lat/lon.
    Returns JSON: {"status":"ok","lat":..., "lon":..., "latency_ms":...}
    If no model is ready, returns status: "model_not_ready"
    """
    start = time.time()
    with raw_buffer_lock:
        if len(raw_buffer) < WINDOW_LENGTH:
            return jsonify({"status": "error", "message": "not enough samples in buffer"}), 400
        # take last WINDOW_LENGTH samples
        last_window = np.stack([x[1] for x in list(raw_buffer)[-WINDOW_LENGTH:]], axis=0)

    with model_lock:
        if (gru_encoder_model is None) or (xgb_model is None):
            return jsonify({"status": "model_not_ready", "message": "model not trained yet"}), 503
        # We need to apply same scaler logic used for training: flatten -> scale -> reshape
        try:
            flat = last_window.reshape(1, -1)
            if os.path.exists(SCALER_SAVE_PATH):
                scaler_local = load(SCALER_SAVE_PATH)
                flat_scaled = scaler_local.transform(flat)
                window_scaled = flat_scaled.reshape(1, WINDOW_LENGTH, SENSOR_DIM)
            else:
                # fallback: standardize per-window
                window_scaled = (last_window - np.mean(last_window, axis=0)) / (np.std(last_window, axis=0) + 1e-6)
                window_scaled = window_scaled.reshape(1, WINDOW_LENGTH, SENSOR_DIM)
            # get embedding
            embedding = gru_encoder_model.predict(window_scaled)
            # predict lat/lon using xgb_model
            pred = xgb_model.predict(embedding)
            lat, lon = float(pred[0][0]), float(pred[0][1])
            latency_ms = int((time.time() - start) * 1000)
            return jsonify({"status": "ok", "lat": lat, "lon": lon, "latency_ms": latency_ms})
        except Exception as e:
            app.logger.exception("Prediction error")
            return jsonify({"status": "error", "message": "prediction failed", "detail": str(e)}), 500

# -----------------------
# Endpoint: GPS retrieval (for your workflow)
# -----------------------
@app.route("/gps", methods=["POST"])
def gps_endpoint():
    """
    Accepts optional JSON {"gps": {"lat":..., "lon":...}}
    If gps provided -> echo back. If not -> return "No signal".
    This design assumes the phone sends its own GPS when hitting this endpoint.
    """
    data = request.get_json(silent=True) or {}
    gps = data.get("gps")
    if gps:
        try:
            lat = float(gps.get("lat"))
            lon = float(gps.get("lon"))
            with latest_gps_lock:
                global latest_gps
                latest_gps = {"lat": lat, "lon": lon, "ts": time.time()}
            return jsonify({"status": "ok", "lat": lat, "lon": lon})
        except Exception:
            return jsonify({"status": "error", "message": "invalid gps format"}), 400
    else:
        # No GPS supplied by phone
        return jsonify({"status": "no_signal", "message": "No signal"}), 204

# -----------------------
# Endpoint: status / debug
# -----------------------
@app.route("/status", methods=["GET"])
def status():
    with raw_buffer_lock, labeled_lock, latest_gps_lock, model_lock:
        return jsonify({
            "buffer_len": len(raw_buffer),
            "labeled_windows": len(labeled_windows),
            "latest_gps": latest_gps,
            "model_ready": (gru_encoder_model is not None and xgb_model is not None)
        })

# -----------------------
# Startup: try loading models and start background trainer
# -----------------------
def start_background_trainer():
    t = threading.Thread(target=background_trainer, daemon=True)
    t.start()
    return t

if __name__ == "__main__":
    # create models folder if missing
    os.makedirs("models", exist_ok=True)

    # Try to load previous models if present
    try:
        loaded_xgb = load_models(MODEL_SAVE_PATH)
        if loaded_xgb is not None:
            xgb_model = loaded_xgb
            app.logger.info("XGBoost loaded from disk.")
    except Exception:
        app.logger.warning("No previous model found or failed to load.")

    # Start background trainer thread
    trainer_thread = start_background_trainer()

    # Run Flask dev server (accessible on LAN)
    app.run(host="0.0.0.0", port=5001, debug=True)
    # When server shuts down:
    stop_background = True
    trainer_thread.join(timeout=2.0)
