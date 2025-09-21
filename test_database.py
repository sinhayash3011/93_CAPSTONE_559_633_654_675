# from database import connect_db, init_db, store_csv_to_db, fetch_dataset_df

# conn = connect_db("ins_project.db")
# init_db(conn)

# # Load your IMU.csv
# store_csv_to_db(conn, "IMU.csv", replace=True)

# # Fetch first 5 rows
# df = fetch_dataset_df(conn, limit=5)
# print(df)



from database import (
    connect_db, init_db, store_csv_to_db, fetch_dataset_df,
    save_model_to_db, list_models
)

# =======================
# 1. Connect & Init DB
# =======================
conn = connect_db("ins_project.db")
init_db(conn)

# =======================
# 2. Store CSV
# =======================
store_csv_to_db(conn, "IMU.csv", replace=True)
print("[INFO] CSV stored into imu_data.")

# Fetch sample rows to verify
df = fetch_dataset_df(conn, limit=5)
print("\n[INFO] Sample imu_data rows:")
print(df)

# =======================
# 3. Store Model
# =======================
model_path = "gps_predictor.h5"  # ensure model file exists
save_model_to_db(conn, model_path, name="gps_predictor", framework="keras")
print(f"\n[INFO] Model '{model_path}' saved into DB.")

# =======================
# 4. Verify Stored Models
# =======================
models = list_models(conn)
print("\n[INFO] Models in database:")
for m in models:
    print(f"- ID: {m['id']}, Name: {m['name']}, Framework: {m['framework']}, Created: {m['created_at']}")

conn.close()
