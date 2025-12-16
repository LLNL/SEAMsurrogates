"""
Borehole synthetic test function subclass and functions for borehole data.
"""

import warnings

import numpy as np
import pandas as pd
import torch

from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, List, Tuple, Union
from scipy.spatial import cKDTree  # type: ignore
from scipy.stats import qmc
from sklearn.model_selection import train_test_split


class Borehole(SyntheticTestFunction):
    """
    Borehole test function.

    This is the 8 dimensional borehole function used as a test case
    in computer experiments. Implementation follows the definition from
    Sonja Surjanovic and Derek Bingham (SFU).

    Inputs (in order):
        rw  : radius of borehole (m)
        r   : radius of influence (m)
        Tu  : transmissivity upper aquifer (m^2/yr)
        Hu  : potentiometric head upper aquifer (m)
        Tl  : transmissivity lower aquifer (m^2/yr)
        Hl  : potentiometric head lower aquifer (m)
        L   : length of borehole (m)
        Kw  : hydraulic conductivity of borehole (m/yr)

    Vector form:
        x = [rw, r, Tu, Hu, Tl, Hl, L, Kw]

    Output:
        y = water flow rate (m^3/yr)

    Reference:
        https://www.sfu.ca/~ssurjano/borehole.html
    """

    _check_grad_at_opt: bool = False

    def __init__(
        self,
        noise_std: Optional[float] = None,
        negate: bool = False,
        bounds: Optional[List[Tuple[float, float]]] = None,
    ) -> None:
        """
        Initialize the Borehole synthetic test function.

        Args:
            noise_std (float or None): Standard deviation of observation noise.
                If None, the function is noise free.
            negate (bool): If True, returns the negative of the standard
                Borehole output, so that the function is maximized at the
                original minimum.
            bounds (list[tuple[float, float]] or None): Optional custom bounds
                as a list of (lower, upper) tuples, one per input dimension,
                in the order documented in the class docstring. If None, uses
                the standard SFU Borehole bounds.
        """

        # Borehole has fixed dimension 8
        self.dim = 8

        # Default bounds from SFU (Surjanovic & Bingham) matching the input order
        if bounds is None:
            bounds = [
                (0.05, 0.15),  # rw
                (100.0, 50000.0),  # r
                (63070.0, 115600.0),  # Tu
                (990.0, 1110.0),  # Hu
                (63.1, 116.0),  # Tl
                (700.0, 820.0),  # Hl
                (1120.0, 1680.0),  # L
                (9855.0, 12045.0),  # Kw
            ]

        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def _evaluate_true(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Evaluate the Borehole test function at given inputs.

        Args:
            X (torch.Tensor or np.ndarray): Input locations, either:
                - 1D tensor/array of shape [8] for a single point, or
                - 2D tensor/array of shape [n, 8] for a batch of n points.

        Returns:
            torch.Tensor: 1D tensor of shape [n] with Borehole function values
            (or shape [1] for a single 1D input). If `self.negate` is True,
            returns the negative of the original Borehole function values.

        Raises:
            TypeError: If `X` is not a `torch.Tensor` or `np.ndarray`.
            ValueError: If the last dimension of `X` is not 8.
        """

        # Convert numpy to torch if needed
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X.astype(np.float32))

        if not isinstance(X, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor or numpy.ndarray.")

        # Ensure 2D: [batch, dim]
        if X.ndim == 1:
            X = X.unsqueeze(0)

        if X.shape[-1] != 8:
            raise ValueError(f"Borehole expects input dimension 8, got {X.shape[-1]}")

        # xx = [rw, r, Tu, Hu, Tl, Hl, L, Kw]
        rw = X[..., 0]
        r = X[..., 1]
        Tu = X[..., 2]
        Hu = X[..., 3]
        Tl = X[..., 4]
        Hl = X[..., 5]
        L = X[..., 6]
        Kw = X[..., 7]

        # SFU implementation
        log_term = torch.log(r / rw)

        frac1 = 2.0 * np.pi * Tu * (Hu - Hl)
        frac2a = 2.0 * L * Tu / (log_term * rw.pow(2) * Kw)
        frac2b = Tu / Tl
        frac2 = log_term * (1.0 + frac2a + frac2b)

        y = frac1 / frac2

        if self.negate:
            y = -y

        return y


def load_data(
    path_to_csv: str = "../../data/borehole_10k.csv",
    n_samples: int = 10000,
    random: bool = True,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load a subset of the borehole dataset from a CSV file.

    Args:
        path_to_csv (str): Path to the CSV file.
        n_samples (int): Number of rows to load. Defaults to dataset size.
        random (bool): If True, select rows randomly. Otherwise, select the
            first n_samples rows.
        seed (int or None): Random seed for reproducibility (used if random is
            True). Defaults to None.

    Returns:
        pd.DataFrame: DataFrame with columns
        ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"].
    """
    df = pd.read_csv(path_to_csv)
    df.columns = ["rw", "r", "Tu", "Hu", "Tl", "Hl", "L", "Kw", "y"]

    # Check and warn if n_samples is too large
    if n_samples > len(df):
        warnings.warn(
            "n_samples is greater than the number of rows in the dataset "
            f"({len(df)}). Using the full dataset instead."
        )
        n_samples = len(df)

    # Select rows
    if random:
        print(
            f"Selecting {n_samples} samples at random from the borehole_10k dataset (seed={seed}).\n"
        )
        df = df.sample(n=n_samples, random_state=seed)
    else:
        print(
            f"Selecting the first {n_samples} samples from the borehole_10k dataset.\n"
        )
        df = df.iloc[:n_samples]

    return df


def split_data(df: pd.DataFrame, LHD: bool = False, n_train: int = 100, seed: int = 42):
    """
    Split data into train and test sets using either Latin Hypercube Design
    (LHD) or random split.

    Args:
        df (pd.DataFrame): Input DataFrame where the last column is the label.
        LHD (bool): If True, use Latin Hypercube Design for selecting training
            samples. If False, use random split. Defaults to False.
        n_train (int): Number of training samples to select. Defaults to 100.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            x_train: Training features array.
            x_test: Testing features array.
            y_train: Training labels array (reshaped to column vector).
            y_test: Testing labels array (reshaped to column vector).

    Raises:
        ValueError: If n_train is greater than the total number of samples in df.
    """
    # Split the data into features (x) and labels (y)
    x = df.iloc[:, :-1].to_numpy()
    y = df.iloc[:, -1].to_numpy()
    n_total, k = x.shape

    # Ensure n_train is not greater than total_samples
    if n_train > n_total:
        raise ValueError(
            f"n_train cannot be greater than the total number of samples "
            f"({n_total})."
        )

    if LHD:
        print(
            "Using n_train closest points to Latin Hypercube Design for"
            " training points.\n"
        )
        # Latin Hypercube Sampling for n_train points in k dimensions
        LHD_gen = qmc.LatinHypercube(d=k, seed=seed)  # type: ignore
        x_lhd = LHD_gen.random(n=n_train)
        # Scale LHD points to the range of x
        for i in range(k):
            x_lhd[:, i] = x_lhd[:, i] * (np.max(x[:, i]) - np.min(x[:, i])) + np.min(
                x[:, i]
            )
        # Build KDTree for nearest neighbor search
        tree = cKDTree(x)

        def query_unique(tree, small_data):
            used_indices = set()
            unique_indices = []
            unique_distances = []

            for point in small_data:
                distances, indices = tree.query(point, k=50)
                for dist, idx in zip(distances, indices):
                    if idx not in used_indices:
                        used_indices.add(idx)
                        unique_indices.append(idx)
                        unique_distances.append(dist)
                        break
            return np.array(unique_distances), np.array(unique_indices)

        # Query for unique nearest neighbors
        distances, index = query_unique(tree, x_lhd)

        x_train = x[index, :]
        y_train = y[index].reshape(-1, 1)
        mask = np.ones(n_total, dtype=bool)
        mask[index] = False
        x_test = x[mask, :]
        y_test = y[mask].reshape(-1, 1)
    else:
        # Standard random split
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            train_size=n_train,
            test_size=None,
            random_state=seed,
        )
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

    print(f"x_train shape: {x_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape} \n")

    return x_train, x_test, y_train, y_test
