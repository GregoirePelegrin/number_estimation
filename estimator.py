import numpy as np


class Estimator:
    @staticmethod
    def get_avg_gap(observations: np.ndarray) -> float:
        gaps: list[int] = []
        current_postition: int = 1

        for observation in observations:
            gaps.append(observation - current_postition - 1)
            current_postition = observation

        return sum(gaps) / len(gaps)

    @staticmethod
    def real_estimate(observations: np.ndarray) -> float:
        return observations.max() + Estimator.get_avg_gap(observations)

    @staticmethod
    def my_estimate(observations: np.ndarray) -> float:
        return 2 * observations.mean()

    @staticmethod
    def minimum_variance_unbiased_estimate(observations: np.ndarray) -> float:
        return (1 + 1./observations.size) * observations.max() - 1
