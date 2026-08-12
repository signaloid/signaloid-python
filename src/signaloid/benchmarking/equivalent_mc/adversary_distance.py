#   Copyright (c) 2026, Signaloid.
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to
#   deal in the Software without restriction, including without limitation the
#   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
#   sell copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in
#   all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.

import multiprocessing
from typing import Callable

import numpy as np
import ot  # type: ignore
from scipy.stats import wasserstein_distance  # type: ignore


def _wasserstein_1_distance(
    adversary_samples: np.ndarray,
    ground_truth_positions: np.ndarray,
    ground_truth_masses: np.ndarray | None,
) -> float:
    if ground_truth_masses is None:
        return float(wasserstein_distance(adversary_samples, ground_truth_positions))
    return float(
        wasserstein_distance(
            u_values=adversary_samples,
            v_values=ground_truth_positions,
            v_weights=ground_truth_masses,
        )
    )


def _wasserstein_2_distance(
    adversary_samples: np.ndarray,
    ground_truth_positions: np.ndarray,
    ground_truth_masses: np.ndarray | None,
) -> float:
    if ground_truth_masses is None:
        return float(
            np.sqrt(
                ot.wasserstein_1d(
                    u_values=adversary_samples, v_values=ground_truth_positions, p=2
                )
            )
        )
    return float(
        np.sqrt(
            ot.wasserstein_1d(
                u_values=adversary_samples,
                v_values=ground_truth_positions,
                v_weights=ground_truth_masses,
                p=2,
            )
        )
    )


def _adversary_distance_loop(
    adversary_size_array: list[int] | np.ndarray,
    adversary_array: np.ndarray,
    ground_truth_positions: np.ndarray,
    progress_queue: "multiprocessing.Queue[int]",
    ground_truth_masses: np.ndarray | None,
    distance_fn: Callable[[np.ndarray, np.ndarray, np.ndarray | None], float],
    seed: int | None = None,
) -> list[tuple[int, list[float]]]:
    """
    Subsample the adversary array at each requested size and score it.

    Shared by the W1/W2 wrappers. Only ``distance_fn`` (the metric) differs.

    Args:
        adversary_size_array: Sample sizes to draw and score, in order.
        adversary_array: Pool of adversary samples to subsample from.
        ground_truth_positions: Ground-truth distribution positions.
        progress_queue: Queue notified as each size is scored, for the
            progress bar.
        ground_truth_masses: Ground-truth masses, or ``None`` for unweighted
            samples.
        distance_fn: Metric applied to (samples, positions, masses).
        seed: RNG seed. A fresh independent seed is drawn when ``None``.

    Returns:
        A list of ``(size, [distance])`` tuples, one per requested size.
    """
    output_list = []
    steps_per_progress_bar_update = 1

    # Generate a new seed for each calculation to achieve independence
    if seed is None:
        seed = np.random.SeedSequence().generate_state(1)[0]
    rng = np.random.default_rng(seed=seed)

    for j, adversary_sample_size in enumerate(adversary_size_array):
        # Subsample with replacement (requested size may exceed the array)
        sample_indices = rng.choice(
            adversary_array.size, size=adversary_sample_size, replace=True
        )
        adversary_samples = adversary_array[sample_indices]

        distance = distance_fn(
            adversary_samples, ground_truth_positions, ground_truth_masses
        )
        output_list.append((adversary_sample_size, [distance]))

        # Update progress bar
        if j % steps_per_progress_bar_update == 0 and progress_queue is not None:
            progress_queue.put(steps_per_progress_bar_update)

    return output_list


def _wasserstein_1_adversary_wrapper(
    adversary_size_array: list[int] | np.ndarray,
    adversary_array: np.ndarray,
    ground_truth_positions: np.ndarray,
    progress_queue: "multiprocessing.Queue[int]",
    ground_truth_masses: np.ndarray | None = None,
    seed: int | None = None,
) -> list[tuple[int, list[float]]]:
    """
    Score adversary subsamples against the ground truth under Wasserstein-1.

    Thin wrapper over :func:`_adversary_distance_loop` with the W1 metric. See
    it for argument and return details.
    """
    return _adversary_distance_loop(
        adversary_size_array,
        adversary_array,
        ground_truth_positions,
        progress_queue,
        ground_truth_masses,
        _wasserstein_1_distance,
        seed=seed,
    )


def _wasserstein_2_adversary_wrapper(
    adversary_size_array: list[int] | np.ndarray,
    adversary_array: np.ndarray,
    ground_truth_positions: np.ndarray,
    progress_queue: "multiprocessing.Queue[int]",
    ground_truth_masses: np.ndarray | None = None,
) -> list[tuple[int, list[float]]]:
    """
    Score adversary subsamples against the ground truth under Wasserstein-2.

    Thin wrapper over :func:`_adversary_distance_loop` with the W2 metric. See
    it for argument and return details.
    """
    return _adversary_distance_loop(
        adversary_size_array,
        adversary_array,
        ground_truth_positions,
        progress_queue,
        ground_truth_masses,
        _wasserstein_2_distance,
    )
