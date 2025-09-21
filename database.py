# database.py
import sqlite3
import pandas as pd
import datetime
import json
from typing import Optional, Dict, Any, List


# ======================
# 1. Connection & Init
# ======================
def connect_db(db_path: str = "ins_project.db") -> sqlite3.Connection:
    """Create/return SQLite connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    """Create tables if not exist."""
    cur = conn.cursor()

    # Table for IMU + GPS data (matches INS.csv exactly)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS imu_data (
        time TEXT,
        ax REAL, ay REAL, az REAL,
        wx REAL, wy REAL, wz REAL,
        Bx REAL, By REAL, Bz REAL,
        latitude REAL, longitude REAL,
        altitude REAL, speed REAL
    );
    """)

    # Table for models
    cur.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT UNIQUE,
        framework TEXT,
        created_at TEXT,
        model_blob BLOB,
        metadata TEXT
    );
    """)

    conn.commit()


# ======================
# 2. Dataset Functions
# ======================
def store_csv_to_db(conn: sqlite3.Connection, csv_path: str, replace: bool = True):
    """Store INS.csv data into imu_data table."""
    df = pd.read_csv(csv_path)

    # Ensure dataset matches expected columns
    expected_cols = [
        "time", "ax", "ay", "az",
        "wx", "wy", "wz",
        "Bx", "By", "Bz",
        "latitude", "longitude", "altitude", "speed"
    ]
    if list(df.columns) != expected_cols:
        raise ValueError(f"CSV columns do not match expected schema.\nExpected: {expected_cols}\nFound: {list(df.columns)}")

    if replace:
        conn.execute("DELETE FROM imu_data;")
        conn.commit()

    df.to_sql("imu_data", conn, if_exists="append", index=False)


def fetch_dataset_df(conn: sqlite3.Connection, limit: Optional[int] = None) -> pd.DataFrame:
    """Fetch dataset from DB into pandas DataFrame."""
    sql = "SELECT * FROM imu_data"
    if limit:
        sql += f" LIMIT {limit}"
    return pd.read_sql(sql, conn, parse_dates=["time"])


# ======================
# 3. Model Functions
# ======================
def save_model_to_db(
    conn: sqlite3.Connection,
    model_filepath: str,
    name: str,
    framework: str = "keras",
    metadata: Optional[Dict[str, Any]] = None
):
    """Save a trained model file (.h5, .pth, etc.) into DB as a BLOB."""
    with open(model_filepath, "rb") as f:
        blob = f.read()

    created_at = datetime.datetime.utcnow().isoformat()
    meta_json = json.dumps(metadata or {})

    cur = conn.cursor()
    cur.execute("DELETE FROM models WHERE name = ?", (name,))
    cur.execute("""
        INSERT INTO models (name, framework, created_at, model_blob, metadata)
        VALUES (?, ?, ?, ?, ?)
    """, (name, framework, created_at, sqlite3.Binary(blob), meta_json))
    conn.commit()


def load_model_from_db(conn: sqlite3.Connection, name: str, out_filepath: Optional[str] = None) -> Optional[str]:
    """Load model BLOB from DB and write to file. Returns filepath."""
    cur = conn.cursor()
    cur.execute("SELECT model_blob, framework FROM models WHERE name = ?", (name,))
    row = cur.fetchone()
    if row is None:
        return None

    blob = row["model_blob"]
    framework = row["framework"].lower()

    if out_filepath is None:
        ext = ".h5" if "keras" in framework or "tensorflow" in framework else ".pth"
        out_filepath = f"{name}_restored{ext}"

    with open(out_filepath, "wb") as f:
        f.write(blob)

    return out_filepath


def list_models(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """List all stored models."""
    cur = conn.cursor()
    cur.execute("SELECT id, name, framework, created_at, metadata FROM models ORDER BY created_at DESC")
    rows = cur.fetchall()
    return [
        {
            "id": r["id"],
            "name": r["name"],
            "framework": r["framework"],
            "created_at": r["created_at"],
            "metadata": json.loads(r["metadata"]) if r["metadata"] else {}
        }
        for r in rows
    ]
