from matplotlib import pyplot as plt
import numpy as np

from estimator import Estimator

NUMBER_RETRIES: int = 1_000


def main():
    numbers_objects: np.ndarray = np.array([30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 1_000_000, 1_000_000_000][:4])

    real_estimation_avg_errors: np.ndarray = np.zeros([len(numbers_objects), 10])
    my_estimation_avg_errors: np.ndarray = np.zeros([len(numbers_objects), 10])
    mvu_estimation_avg_errors: np.ndarray = np.zeros([len(numbers_objects), 10])

    for i1, number_objects in enumerate(numbers_objects):
        print(f"Current number of objects: {number_objects}")
        objects: np.ndarray = np.arange(number_objects)
        numbers_observations: list[int] = [int(0.1 * (i + 1) * number_objects) for i in range(10)]
        for i2, number_observations in enumerate(numbers_observations):
            real_estimation_errors: list[float] = []
            my_estimation_errors: list[float] = []
            mvu_estimation_errors: list[float] = []
            for _ in range(NUMBER_RETRIES):
                observations: np.ndarray = np.random.choice(objects, size=number_observations, replace=False)

                real_estimation: float = Estimator.real_estimate(observations=observations)
                my_estimation: float = Estimator.my_estimate(observations=observations)
                mvu_estimation: float = Estimator.minimum_variance_unbiased_estimate(observations=observations)

                real_estimation_errors.append(abs(number_objects - real_estimation) / number_objects * 100)
                my_estimation_errors.append(abs(number_objects - my_estimation) / number_objects * 100)
                mvu_estimation_errors.append(abs(number_objects - mvu_estimation) / number_objects * 100)
            real_estimation_avg_errors[i1, i2] = sum(real_estimation_errors) / NUMBER_RETRIES
            my_estimation_avg_errors[i1, i2] = sum(my_estimation_errors) / NUMBER_RETRIES
            mvu_estimation_avg_errors[i1, i2] = sum(mvu_estimation_errors) / NUMBER_RETRIES

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.set_xlim(0.9, 1.1)
    ax.set_ylim(0, max(real_estimation_avg_errors.max(), my_estimation_avg_errors.max()))
    ax.set_zlim(-1, len(numbers_objects))
    ax.set_xlabel("#observations / #total_objects")
    ax.set_ylabel("normed average error over 1_000 tries")
    ax.set_zlabel("#total_objects")

    x = np.array([.1 * (i + 1) for i in range(10)])
    ax.set_xticks(x)
    ax.set_zticks([i for i in range(len(numbers_objects))], labels=numbers_objects)

    for i in range(len(numbers_objects)):
        y1 = real_estimation_avg_errors[i]
        y2 = my_estimation_avg_errors[i]
        y3 = mvu_estimation_avg_errors[i]

        ax.plot(x, y1, zs=i, zdir="z", marker="o", color=(1, 0, 0, 0.5))
        ax.plot(x, y2, zs=i, zdir="z", marker="o", color=(0, 0, 1, 0.5))
        ax.plot(x, y3, zs=i, zdir="z", marker="o", color=(0, 1, 0, 0.5))

    plt.show()


if __name__ == "__main__":
    main()
