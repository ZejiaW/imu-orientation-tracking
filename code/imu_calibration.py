import numpy as np
import matplotlib.pyplot as plt


def estimate_bias(imu_data, scale_factor_accz, pre=100):
    """
    Using the first [pre] samples to estimate the bias of the IMU.
    IMU data should be a Numpy array of shape (7, N) where N is the number of samples.
    Each row of the IMU data is: [timestamp, AccX, AccY, AccZ, GyrX, GyrY, GyrZ]
    Returns a Numpy array of shape (6,) with the estimated bias.

    bias = raw - value / scale
    """
    bias = np.mean(imu_data[1:, :pre], axis=1)
    # for AccZ we should consider the gravity
    bias[2] -= 1.0 / scale_factor_accz
    print(f"Estimated bias: x={bias[0]:.4f}, y={bias[1]:.4f}, z={bias[2]:.4f}, gx={bias[3]:.4f}, gy={bias[4]:.4f}, gz={bias[5]:.4f}")
    return bias

def raw2physical(raw, sensitivity, Vref=3300, bias_pre=100, copy=True):
    """
    Convert raw IMU data to physical units.
    raw: Numpy array of shape (7, N) where N is the number of samples.
    sensitivity: Numpy array of shape (6,) with the sensitivity of the IMU.
    Vref: reference voltage of the IMU.
    bias: Numpy array of shape (6,) with the estimated bias.
    Returns a Numpy array of shape (7, N) with the physical units.

    physical = (raw - bias) * (Vref / 1023 / sensitivity)
    """
    if copy:
        data = raw.copy()
    else:
        data = raw

    # data[1:4, :] = - data[1:4, :]
    scale_factors = Vref / 1023 / sensitivity
    bias = estimate_bias(data, scale_factors[2], pre=bias_pre)
    data[1:4, :] = (data[1:4, :] - bias[:3, None]) * scale_factors[:3, None]
    data[4:, :] = (data[4:, :] - bias[3:, None]) * scale_factors[3:, None] * np.pi / 180
    return data