import numpy as np

class SimpleSSLTransform:
    def __init__(
        self,
        flip_prob: float = 0.5,
        noise_prob: float = 0.3,
        noise_std: float = 0.01,
    ) -> None:
        self.flip_prob = flip_prob
        self.noise_prob = noise_prob
        self.noise_std = noise_std

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # x shape: [C, H, W, D]
        x = x.copy()

        # random flip
        if np.random.rand() < self.flip_prob:
            axis = np.random.choice([1, 2, 3])  # H/W/D axes
            x = np.flip(x, axis=axis).copy()

        # gaussian noise
        if np.random.rand() < self.noise_prob:
            noise = np.random.normal(0, self.noise_std, size=x.shape).astype(np.float32)
            x = x + noise

        # intensity shift (scale and offset) - Phase 15.5 Fix
        if np.random.rand() < 0.5:
            scale = np.random.uniform(0.95, 1.05)
            offset = np.random.uniform(-0.05, 0.05)
            x = (x * scale) + offset

        return x.astype(np.float32)
