import numpy as np

def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    v = np.random.rand(data.shape[1])
    for i in range(num_steps):
        v_prev = v
        v = data.dot(v) / np.linalg.norm(data.dot(v_prev))
        lambda_x = v_prev.dot(data.dot(v)) / v_prev.dot(v_prev)
        
    return float(lambda_x), v