import numpy as np
import torch
from botorch.test_functions.synthetic import SyntheticTestFunction
from typing import Optional, List, Tuple, Union


class Borehole(SyntheticTestFunction):
    """
    Borehole test function.

    This is the 8 dimensional borehole function used as a test case
    in computer experiments. Implementation follows the definition from
    Sonja Surjanovic and Derek Bingham (SFU).

    Inputs:
        x = [rw, r, Tu, Hu, Tl, Hl, L, Kw]

    Output:
        y  = water flow rate (m^3/yr)

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
        Initialize Borehole test function.

        Args:
            noise_std: Standard deviation of observation noise.
            negate: If True, returns the negative of the standard borehole output.
            bounds: Optional custom bounds as list of (lower, upper) tuples
                for each of the 8 inputs. If None, defaults to the standard
                SFU borehole bounds.
        """

        # Borehole has fixed dimension 8
        self.dim = 8

        # Default bounds from SFU (Surjanovic & Bingham) in the same order
        # xx = [rw, r, Tu, Hu, Tl, Hl, L, Kw]
        if bounds is None:
            bounds = [
                (0.05, 0.15),        # rw  radius of borehole (m)
                (100.0, 50000.0),    # r   radius of influence (m)
                (63070.0, 115600.0), # Tu  transmissivity upper aquifer (m^2/yr)
                (990.0, 1110.0),     # Hu  potentiometric head upper aquifer (m)
                (63.1, 116.0),       # Tl  transmissivity lower aquifer (m^2/yr)
                (700.0, 820.0),      # Hl  potentiometric head lower aquifer (m)
                (1120.0, 1680.0),    # L   length of borehole (m)
                (9855.0, 12045.0),   # Kw  hydraulic conductivity of borehole (m/yr)
            ]

        self.continuous_inds = list(range(self.dim))
        self.discrete_inds = []
        self.categorical_inds = []

        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)

    def _evaluate_true(self, X: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """
        Evaluate the Borehole test function.

        Args:
            X: Input locations, either:
                - 1D tensor/array of shape [8], or
                - 2D tensor/array of shape [n, 8].

        Returns:
            Tensor of shape [n] with Borehole function values (or [1] for a single
            1D input). If `self.negate` is True, returns the negative of original
            borehole funciton values.

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