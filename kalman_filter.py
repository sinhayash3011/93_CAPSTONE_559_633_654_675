# import numpy as np
# from filterpy.kalman import KalmanFilter

# class INS_KalmanFilter:
#     def __init__(self, dt=0.1, process_variance=1e-3, measurement_variance=1e-2):
#         """
#         Kalman Filter for Position + Orientation Estimation using GPS, IMU, and Magnetometer.
#         - Handles cases where GPS is missing during testing (switches to dead reckoning).
#         """
#         self.kf = KalmanFilter(dim_x=9, dim_z=10)  # 9 states, 10 measurements

#         # **State Vector: [x, y, z, vx, vy, vz, roll, pitch, yaw]**
#         # **Measurement: [lat, lon, alt, speed, wx, wy, wz, Bx, By, Bz]**

#         # **State Transition Matrix (Includes Orientation)**
#         self.kf.F = np.eye(9)
#         self.kf.F[0, 3] = self.kf.F[1, 4] = self.kf.F[2, 5] = dt  # Position updates using velocity

#         # **Measurement Function (GPS, Gyro, Magnetometer)**
#         self.kf.H = np.zeros((10, 9))
#         self.kf.H[:4, :4] = np.eye(4)  # GPS updates position & speed
#         self.kf.H[4:7, 6:9] = np.eye(3)  # Gyro updates roll, pitch, yaw
#         self.kf.H[7:10, 6:9] = np.eye(3)  # Magnetometer helps with orientation

#         # **Process Noise Covariance**
#         self.kf.Q = np.eye(9) * process_variance # Models random changes in motion

#         # **Measurement Noise Covariance**
#         self.kf.R = np.eye(10) * measurement_variance # Models sensor uncertainity

#         # **Initial Covariance Matrix**
#         self.kf.P *= 1

#         # **Initial State: Assume rest at origin**
#         self.kf.x = np.zeros((9, 1))



#     def predict(self, ax=0, ay=0, az=0, wx=0, wy=0, wz=0):
#         """Predict Step: Uses accelerometer & gyroscope for dead reckoning"""
#         dt = self.kf.F[0, 3]
        
#         # Update velocities using acceleration
#         self.kf.x[3] += ax * dt
#         self.kf.x[4] += ay * dt
#         self.kf.x[5] += az * dt

#         # Update orientation using gyroscope
#         self.kf.x[6] += wx * dt  # Roll
#         self.kf.x[7] += wy * dt  # Pitch
#         self.kf.x[8] += wz * dt  # Yaw
        
#         self.kf.predict()



#     def update(self, gps_data=None, imu_data=None, mag_data=None):
#         """Update Step: Uses GPS when available, otherwise relies on IMU."""
#         if gps_data is not None:  # If GPS is available, update normally
#             measurement = np.array(gps_data + imu_data + mag_data).reshape(10, 1)
#             self.kf.update(measurement) # Updating by Updating its kalman gain to decide whether to decide more on sensor data or prediction calculation
#         else:  # If GPS is missing, skip update step and keep predicting
#             pass



#     def get_state(self):
#         """Returns estimated state: [x, y, z, vx, vy, vz, roll, pitch, yaw]"""
#         return self.kf.x.flatten()





######################################################################################################################################################################


# kalman_filter.py
import numpy as np
import pandas as pd

class SimpleKalmanFilter:
    def __init__(self, state_dim=4, meas_dim=2, process_var=1e-3, meas_var=1e-2):
        """
        state_dim = [x, y, vx, vy]
        meas_dim = [latitude, longitude] (treated as position measurements)
        """
        self.state_dim = state_dim
        self.meas_dim = meas_dim

        # State vector [x, y, vx, vy]
        self.x = np.zeros((state_dim, 1))

        # State covariance
        self.P = np.eye(state_dim)

        # Transition matrix (constant velocity model)
        self.F = np.eye(state_dim)

        # Measurement matrix (we measure x, y only)
        self.H = np.zeros((meas_dim, state_dim))
        self.H[0, 0] = 1
        self.H[1, 1] = 1

        # Process & measurement noise
        self.Q = process_var * np.eye(state_dim)
        self.R = meas_var * np.eye(meas_dim)

    def predict(self, dt=1.0):
        """
        Predict next state using constant velocity model.
        """
        # Update F with dt
        self.F[0, 2] = dt
        self.F[1, 3] = dt

        # Predict state and covariance
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """
        z: measurement [lat, lon]
        """
        z = np.reshape(z, (self.meas_dim, 1))
        y = z - np.dot(self.H, self.x)  # innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Kalman gain
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)
        return self.x


def apply_kalman_filter(df, dt=1.0):
    """
    Apply Kalman filter to dataset.
    Input:
        df → DataFrame with latitude, longitude
    Output:
        filtered_positions → np.array of filtered [lat, lon]
    """

    kf = SimpleKalmanFilter()
    filtered_positions = []

    for i in range(len(df)):
        # Predict step
        kf.predict(dt=dt)

        # Measurement update (use GPS if available)
        z = [df.iloc[i]["latitude"], df.iloc[i]["longitude"]]
        kf.update(z)

        filtered_positions.append([kf.x[0, 0], kf.x[1, 0]])

    return np.array(filtered_positions)


if __name__ == "__main__":
    from dataset_loader import load_dataset

    dataset_path = "IMU.csv"  # replace with your dataset path
    df = load_dataset(dataset_path)

    filtered_positions = apply_kalman_filter(df)

    print("Original GPS (first 5):")
    print(df[["latitude", "longitude"]].head())

    print("\nFiltered GPS (first 5):")
    print(filtered_positions[:5])
