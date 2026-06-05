"""
================================================================================
UNIFIED LINEAR-IN-LEARNABLES (LiL) BASIS FUNCTIONS
================================================================================

A cohesive framework for LiL basis functions including:
    - Chebyshev polynomials
    - Fourier series (cos, sin, or mixed)
    - Extreme Learning Machine (ELM)
    - Tensor products of any 1D bases
    - Augmented bases with polynomial lift

Supports arbitrary combinations like Chebyshev(x) × Sin(y) for mixed problems.

Usage::

    from lilq.basis import Chebyshev1D, Fourier1D, TensorProductBasis2D, ELMBasis2D

    # Single basis type
    basis = TensorProductBasis2D(
        Chebyshev1D(order=20, domain=(0, 1)),
        Chebyshev1D(order=20, domain=(0, 1))
    )

    # Or use the generic factory
    basis = create_basis_2d('fourier', 20, 20, (0, 1), (0, 1))

    # Evaluate and differentiate
    Phi = basis.evaluate(x, y)
    dPhi_dx = basis.derivative(x, y, dx=1, dy=0)

================================================================================
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union


# ==============================================================================
#                           ABSTRACT BASE CLASS
# ==============================================================================

class Basis1D(ABC):
    """Abstract base class for 1D basis functions."""

    @property
    @abstractmethod
    def n_basis(self) -> int:
        """Number of basis functions."""
        pass

    @property
    @abstractmethod
    def domain(self) -> Tuple[float, float]:
        """Physical domain [a, b]."""
        pass

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at points x.

        Parameters
        ----------
        x : array of shape (n_points,)
            Evaluation points.

        Returns
        -------
        Phi : array of shape (n_points, n_basis)
            Basis function values.
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        """Evaluate derivatives of all basis functions at points x.

        Parameters
        ----------
        x : array of shape (n_points,)
            Evaluation points.
        order : int
            Derivative order (1 or 2).

        Returns
        -------
        dPhi : array of shape (n_points, n_basis)
            Derivative values.
        """
        pass

    def __repr__(self):
        return f"{self.__class__.__name__}(n_basis={self.n_basis}, domain={self.domain})"


# ==============================================================================
#                           CHEBYSHEV BASIS
# ==============================================================================

class Chebyshev1D(Basis1D):
    """Chebyshev polynomial basis T_n(x) on domain [a, b].

    Maps physical domain [a, b] to standard domain [-1, 1] and uses
    the recurrence relation for stable evaluation.

    Includes T_0 = 1 (constant) and T_1 = x (linear) automatically.

    Parameters
    ----------
    order : int
        Maximum polynomial order (n = 0, 1, ..., order).
    domain : tuple
        Physical domain [a, b].
    """

    def __init__(self, order: int, domain: Tuple[float, float] = (-1.0, 1.0)):
        self._order = order
        self._domain = (float(domain[0]), float(domain[1]))
        self._a = self._domain[0]
        self._b = self._domain[1]
        self._scale = 2.0 / (self._b - self._a)

    @property
    def n_basis(self) -> int:
        return self._order + 1

    @property
    def domain(self) -> Tuple[float, float]:
        return self._domain

    @property
    def order(self) -> int:
        return self._order

    def _map_to_standard(self, x: np.ndarray) -> np.ndarray:
        """Map from physical [a, b] to standard [-1, 1]."""
        return 2.0 * (x - self._a) / (self._b - self._a) - 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        xi = self._map_to_standard(x)
        n_pts = len(xi)

        T = np.zeros((n_pts, self.n_basis), dtype=np.float64)
        T[:, 0] = 1.0
        if self._order >= 1:
            T[:, 1] = xi
        for n in range(1, self._order):
            T[:, n + 1] = 2.0 * xi * T[:, n] - T[:, n - 1]

        return T

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        xi = self._map_to_standard(x)
        n_pts = len(xi)

        if order == 0:
            return self.evaluate(x)

        elif order == 1:
            T = self.evaluate(x)
            D = np.zeros((n_pts, self.n_basis), dtype=np.float64)
            D[:, 0] = 0.0
            if self._order >= 1:
                D[:, 1] = 1.0
            for n in range(1, self._order):
                D[:, n + 1] = 2.0 * T[:, n] + 2.0 * xi * D[:, n] - D[:, n - 1]
            return D * self._scale

        elif order == 2:
            T = self.evaluate(x)
            D = np.zeros((n_pts, self.n_basis), dtype=np.float64)
            D[:, 0] = 0.0
            if self._order >= 1:
                D[:, 1] = 1.0
            for n in range(1, self._order):
                D[:, n + 1] = 2.0 * T[:, n] + 2.0 * xi * D[:, n] - D[:, n - 1]

            E = np.zeros((n_pts, self.n_basis), dtype=np.float64)
            E[:, 0] = 0.0
            if self._order >= 1:
                E[:, 1] = 0.0
            for n in range(1, self._order):
                E[:, n + 1] = 4.0 * D[:, n] + 2.0 * xi * E[:, n] - E[:, n - 1]
            return E * (self._scale ** 2)

        else:
            raise ValueError(f"Derivative order {order} not implemented (max 2)")


# ==============================================================================
#                           FOURIER BASIS
# ==============================================================================

class Fourier1D(Basis1D):
    """Fourier basis functions on domain [a, b].

    Modes
    -----
    'cos' : cos(n·π·ξ) for n = 0, 1, ..., n_modes-1.
        Note: n=0 gives constant term = 1.
        Automatically satisfies Neumann BCs.
    'sin' : sin(n·π·ξ) for n = 1, 2, ..., n_modes.
        No constant term. Automatically satisfies homogeneous Dirichlet BCs.
    'both' : cos modes followed by sin modes (excluding sin(0)=0).

    Parameters
    ----------
    n_modes : int
        Number of modes (interpretation depends on mode type).
    domain : tuple
        Physical domain [a, b].
    mode : str
        'cos', 'sin', or 'both'.
    """

    def __init__(self, n_modes: int, domain: Tuple[float, float] = (0.0, 1.0),
                 mode: str = 'cos'):
        self._n_modes = n_modes
        self._domain = (float(domain[0]), float(domain[1]))
        self._mode = mode
        self._a = self._domain[0]
        self._b = self._domain[1]
        self._L = self._b - self._a

        if mode == 'cos':
            self._cos_modes = np.arange(n_modes)
            self._sin_modes = np.array([], dtype=int)
        elif mode == 'sin':
            self._cos_modes = np.array([], dtype=int)
            self._sin_modes = np.arange(1, n_modes + 1)
        elif mode == 'both':
            self._cos_modes = np.arange(int((n_modes - 1) / 2) + 1)
            self._sin_modes = np.arange(1, n_modes - len(self._cos_modes) + 1)
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'cos', 'sin', or 'both'.")

    @property
    def n_basis(self) -> int:
        return len(self._cos_modes) + len(self._sin_modes)

    @property
    def domain(self) -> Tuple[float, float]:
        return self._domain

    @property
    def mode(self) -> str:
        return self._mode

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n_pts = len(x)
        xi = (x - self._a) / self._L

        Phi = np.zeros((n_pts, self.n_basis), dtype=np.float64)
        idx = 0

        for n in self._cos_modes:
            Phi[:, idx] = np.cos(n * np.pi * xi)
            idx += 1
        for n in self._sin_modes:
            Phi[:, idx] = np.sin(n * np.pi * xi)
            idx += 1

        return Phi

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n_pts = len(x)

        if order == 0:
            return self.evaluate(x)

        xi = (x - self._a) / self._L
        scale = np.pi / self._L

        dPhi = np.zeros((n_pts, self.n_basis), dtype=np.float64)
        idx = 0

        if order == 1:
            for n in self._cos_modes:
                dPhi[:, idx] = -n * scale * np.sin(n * np.pi * xi)
                idx += 1
            for n in self._sin_modes:
                dPhi[:, idx] = n * scale * np.cos(n * np.pi * xi)
                idx += 1

        elif order == 2:
            for n in self._cos_modes:
                dPhi[:, idx] = -(n * scale) ** 2 * np.cos(n * np.pi * xi)
                idx += 1
            for n in self._sin_modes:
                dPhi[:, idx] = -(n * scale) ** 2 * np.sin(n * np.pi * xi)
                idx += 1

        else:
            raise ValueError(f"Derivative order {order} not implemented (max 2)")

        return dPhi


# ==============================================================================
#                        TENSOR PRODUCT BASIS (2D)
# ==============================================================================

class TensorProductBasis2D:
    """Tensor product of two 1D bases for 2D problems.

    Phi_{nm}(x, y) = phi^x_n(x) · phi^y_m(y)

    Coefficients are ordered row-major: x-index is outer, y-index is inner.

    Parameters
    ----------
    basis_x : Basis1D
        Basis for x-direction.
    basis_y : Basis1D
        Basis for y-direction.
    """

    def __init__(self, basis_x: Basis1D, basis_y: Basis1D):
        self.basis_x = basis_x
        self.basis_y = basis_y
        self._n_basis_x = basis_x.n_basis
        self._n_basis_y = basis_y.n_basis

    @property
    def n_basis(self) -> int:
        return self._n_basis_x * self._n_basis_y

    @property
    def n_basis_x(self) -> int:
        return self._n_basis_x

    @property
    def n_basis_y(self) -> int:
        return self._n_basis_y

    @property
    def domain_x(self) -> Tuple[float, float]:
        return self.basis_x.domain

    @property
    def domain_y(self) -> Tuple[float, float]:
        return self.basis_y.domain

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate all 2D basis functions at points (x, y).

        Returns
        -------
        Phi : array of shape (n_points, n_basis)
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        n_pts = len(x)

        Phi_x = self.basis_x.evaluate(x)
        Phi_y = self.basis_y.evaluate(y)

        Phi = (Phi_x[:, :, np.newaxis] * Phi_y[:, np.newaxis, :]).reshape(n_pts, -1)
        return Phi

    def derivative(self, x: np.ndarray, y: np.ndarray,
                   dx: int = 0, dy: int = 0) -> np.ndarray:
        """Evaluate derivatives d^(dx+dy)Phi / dx^dx dy^dy.

        Returns
        -------
        dPhi : array of shape (n_points, n_basis)
        """
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))
        n_pts = len(x)

        Phi_x = self.basis_x.derivative(x, order=dx) if dx > 0 else self.basis_x.evaluate(x)
        Phi_y = self.basis_y.derivative(y, order=dy) if dy > 0 else self.basis_y.evaluate(y)

        dPhi = (Phi_x[:, :, np.newaxis] * Phi_y[:, np.newaxis, :]).reshape(n_pts, -1)
        return dPhi

    def reconstruct(self, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Reconstruct f(x, y) = Σ c_nm · Phi_nm(x, y)."""
        return self.evaluate(x, y) @ coeffs

    def reconstruct_derivative(self, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray,
                               dx: int = 0, dy: int = 0) -> np.ndarray:
        """Reconstruct derivative from coefficients."""
        return self.derivative(x, y, dx=dx, dy=dy) @ coeffs

    def __repr__(self):
        return (f"TensorProductBasis2D(\n"
                f"  x: {self.basis_x},\n"
                f"  y: {self.basis_y},\n"
                f"  total: {self.n_basis} basis functions\n)")


# ==============================================================================
#                           ELM BASIS (2D)
# ==============================================================================

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None


class ELMBasis2D_Xavier:
    """Extreme Learning Machine basis for 2D problems (Xavier initialization).

    Each basis function is:
        phi_j(x, y) = activation(alpha_j · x_norm + beta_j · y_norm + gamma_j)

    where x_norm, y_norm are normalized to [-1, 1].
    Uses PyTorch autograd for derivatives.

    Parameters
    ----------
    n_hidden : int
        Number of hidden neurons (basis functions).
    domain_x, domain_y : tuple
        Physical domains.
    activation : str
        'tanh' or 'sigmoid'.
    seed : int, optional
        Random seed for reproducibility.
    """

    def __init__(self, n_hidden: int,
                 domain_x: Tuple[float, float],
                 domain_y: Tuple[float, float],
                 activation: str = 'tanh',
                 seed: Optional[int] = 42,
                 dtype=None,
                 device: str = 'cpu'):

        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ELMBasis2D.")

        if dtype is None:
            dtype = torch.float64

        self._n_hidden = n_hidden
        self._domain_x = (float(domain_x[0]), float(domain_x[1]))
        self._domain_y = (float(domain_y[0]), float(domain_y[1]))
        self._activation = activation
        self._dtype = dtype
        self._device = device

        self._ax, self._bx = self._domain_x
        self._ay, self._by = self._domain_y
        self._Lx = self._bx - self._ax
        self._Ly = self._by - self._ay

        # Xavier/Glorot initialization
        if seed is not None:
            torch.manual_seed(seed)

        fan_in = 2
        fan_out = n_hidden
        limit = np.sqrt(6.0 / (fan_in + fan_out))

        self.alpha = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-limit, limit)
        self.beta = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-limit, limit)
        self.gamma = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-limit, limit)

    @property
    def n_basis(self) -> int:
        return self._n_hidden

    @property
    def domain_x(self) -> Tuple[float, float]:
        return self._domain_x

    @property
    def domain_y(self) -> Tuple[float, float]:
        return self._domain_y

    def _normalize(self, x, y):
        """Normalize coordinates to [-1, 1]."""
        x_norm = 2.0 * (x - (self._ax + self._bx) / 2) / self._Lx
        y_norm = 2.0 * (y - (self._ay + self._by) / 2) / self._Ly
        return x_norm, y_norm

    def _forward(self, x, y):
        """Forward pass computing basis function values."""
        x_norm, y_norm = self._normalize(x, y)
        z = x_norm.unsqueeze(1) * self.alpha + y_norm.unsqueeze(1) * self.beta + self.gamma

        if self._activation == 'tanh':
            return torch.tanh(z)
        elif self._activation == 'sigmoid':
            return torch.sigmoid(z)
        else:
            raise ValueError(f"Unknown activation: {self._activation}")

    def _to_tensor(self, arr: np.ndarray, requires_grad: bool = False):
        return torch.tensor(arr, dtype=self._dtype, device=self._device, requires_grad=requires_grad)

    def _to_numpy(self, tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    def evaluate(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Evaluate all basis functions at points (x, y)."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        with torch.no_grad():
            Phi = self._forward(self._to_tensor(x), self._to_tensor(y))
        return self._to_numpy(Phi)

    def derivative(self, x: np.ndarray, y: np.ndarray,
                   dx: int = 0, dy: int = 0) -> np.ndarray:
        """Evaluate derivatives using autograd."""
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        y = np.atleast_1d(np.asarray(y, dtype=np.float64))

        if dx == 0 and dy == 0:
            return self.evaluate(x, y)

        x_t = self._to_tensor(x, requires_grad=True)
        y_t = self._to_tensor(y, requires_grad=True)
        Phi = self._forward(x_t, y_t)

        n_pts = len(x)
        n_hidden = self._n_hidden

        if dx == 1 and dy == 0:
            dPhi = torch.zeros(n_pts, n_hidden, dtype=self._dtype, device=self._device)
            for j in range(n_hidden):
                grad = torch.autograd.grad(Phi[:, j].sum(), x_t, create_graph=True)[0]
                dPhi[:, j] = grad
            return self._to_numpy(dPhi)

        elif dx == 0 and dy == 1:
            dPhi = torch.zeros(n_pts, n_hidden, dtype=self._dtype, device=self._device)
            for j in range(n_hidden):
                grad = torch.autograd.grad(Phi[:, j].sum(), y_t, create_graph=True)[0]
                dPhi[:, j] = grad
            return self._to_numpy(dPhi)

        elif dx == 2 and dy == 0:
            d2Phi = torch.zeros(n_pts, n_hidden, dtype=self._dtype, device=self._device)
            for j in range(n_hidden):
                grad1 = torch.autograd.grad(Phi[:, j].sum(), x_t, create_graph=True)[0]
                grad2 = torch.autograd.grad(grad1.sum(), x_t, retain_graph=True)[0]
                d2Phi[:, j] = grad2
            return self._to_numpy(d2Phi)

        elif dx == 0 and dy == 2:
            d2Phi = torch.zeros(n_pts, n_hidden, dtype=self._dtype, device=self._device)
            for j in range(n_hidden):
                grad1 = torch.autograd.grad(Phi[:, j].sum(), y_t, create_graph=True)[0]
                grad2 = torch.autograd.grad(grad1.sum(), y_t, retain_graph=True)[0]
                d2Phi[:, j] = grad2
            return self._to_numpy(d2Phi)

        elif dx == 1 and dy == 1:
            d2Phi = torch.zeros(n_pts, n_hidden, dtype=self._dtype, device=self._device)
            for j in range(n_hidden):
                grad_x = torch.autograd.grad(Phi[:, j].sum(), x_t, create_graph=True)[0]
                grad_xy = torch.autograd.grad(grad_x.sum(), y_t, retain_graph=True)[0]
                d2Phi[:, j] = grad_xy
            return self._to_numpy(d2Phi)

        else:
            raise ValueError(f"Derivative order (dx={dx}, dy={dy}) not implemented")

    def reconstruct(self, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Reconstruct function from coefficients."""
        return self.evaluate(x, y) @ coeffs

    def reconstruct_derivative(self, coeffs: np.ndarray, x: np.ndarray, y: np.ndarray,
                               dx: int = 0, dy: int = 0) -> np.ndarray:
        """Reconstruct derivative from coefficients."""
        return self.derivative(x, y, dx=dx, dy=dy) @ coeffs

    def __repr__(self):
        return (f"ELMBasis2D(n_hidden={self._n_hidden}, "
                f"domain_x={self._domain_x}, domain_y={self._domain_y}, "
                f"activation='{self._activation}')")


class ELMBasis2D(ELMBasis2D_Xavier):
    """ELM basis with Kaiming uniform initialization matching nn.Linear defaults.

    PyTorch nn.Linear uses:
        weights: uniform(-bound, bound) where bound = sqrt(3 / fan_in)
        biases:  uniform(-bound, bound) where bound = 1 / sqrt(fan_in)

    For the ELM, fan_in = 2 (x and y inputs), so:
        alpha, beta (weight-like): bound = sqrt(3/2) ≈ 1.2247
        gamma (bias-like):         bound = 1/sqrt(2)  ≈ 0.7071
    """

    def __init__(self, n_hidden: int,
                 domain_x: Tuple[float, float],
                 domain_y: Tuple[float, float],
                 activation: str = 'tanh',
                 seed: Optional[int] = 42,
                 dtype=None,
                 device: str = 'cpu'):

        if dtype is None:
            dtype = torch.float64

        self._n_hidden = n_hidden
        self._domain_x = (float(domain_x[0]), float(domain_x[1]))
        self._domain_y = (float(domain_y[0]), float(domain_y[1]))
        self._activation = activation
        self._dtype = dtype
        self._device = device

        self._ax, self._bx = self._domain_x
        self._ay, self._by = self._domain_y
        self._Lx = self._bx - self._ax
        self._Ly = self._by - self._ay

        if seed is not None:
            torch.manual_seed(seed)

        fan_in = 2
        weight_bound = np.sqrt(3.0 / fan_in)
        bias_bound = 1.0 / np.sqrt(fan_in)

        self.alpha = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-weight_bound, weight_bound)
        self.beta = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-weight_bound, weight_bound)
        self.gamma = torch.empty(n_hidden, dtype=dtype, device=device).uniform_(-bias_bound, bias_bound)


# ==============================================================================
#                      AUGMENTED BASIS (with polynomial lift)
# ==============================================================================

class AugmentedBasis1D(Basis1D):
    """Augmented basis with explicit polynomial terms.

    f(x) = a_0 + a_1·x + Σ c_j·phi_j(x)

    Ensures constant and linear modes are always present,
    regardless of the underlying basis type.

    Parameters
    ----------
    base_basis : Basis1D
        The underlying basis (Fourier, etc.).
    include_constant : bool
        Whether to add explicit constant term.
    include_linear : bool
        Whether to add explicit linear term.
    """

    def __init__(self, base_basis: Basis1D,
                 include_constant: bool = True,
                 include_linear: bool = True):
        self._base = base_basis
        self._include_constant = include_constant
        self._include_linear = include_linear
        self._n_augment = int(include_constant) + int(include_linear)

    @property
    def n_basis(self) -> int:
        return self._n_augment + self._base.n_basis

    @property
    def domain(self) -> Tuple[float, float]:
        return self._base.domain

    def _normalized_x(self, x: np.ndarray) -> np.ndarray:
        """Normalize x to [-1, 1] for the linear term."""
        a, b = self.domain
        return 2.0 * (x - a) / (b - a) - 1.0

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n_pts = len(x)

        Phi = np.zeros((n_pts, self.n_basis), dtype=np.float64)
        idx = 0

        if self._include_constant:
            Phi[:, idx] = 1.0
            idx += 1
        if self._include_linear:
            Phi[:, idx] = self._normalized_x(x)
            idx += 1

        Phi[:, idx:] = self._base.evaluate(x)
        return Phi

    def derivative(self, x: np.ndarray, order: int = 1) -> np.ndarray:
        x = np.atleast_1d(np.asarray(x, dtype=np.float64))
        n_pts = len(x)

        if order == 0:
            return self.evaluate(x)

        dPhi = np.zeros((n_pts, self.n_basis), dtype=np.float64)
        idx = 0

        a, b = self.domain
        scale = 2.0 / (b - a)

        if self._include_constant:
            dPhi[:, idx] = 0.0
            idx += 1
        if self._include_linear:
            dPhi[:, idx] = scale if order == 1 else 0.0
            idx += 1

        dPhi[:, idx:] = self._base.derivative(x, order=order)
        return dPhi


# ==============================================================================
#                       CONVENIENCE FACTORY FUNCTIONS
# ==============================================================================

def create_chebyshev_basis_2d(order_x: int, order_y: int,
                              domain_x: Tuple[float, float],
                              domain_y: Tuple[float, float]) -> TensorProductBasis2D:
    """Create a Chebyshev tensor product basis."""
    return TensorProductBasis2D(
        Chebyshev1D(order_x, domain_x),
        Chebyshev1D(order_y, domain_y)
    )


def create_fourier_basis_2d(n_modes_x: int, n_modes_y: int,
                            domain_x: Tuple[float, float],
                            domain_y: Tuple[float, float],
                            mode_x: str = 'cos',
                            mode_y: str = 'cos') -> TensorProductBasis2D:
    """Create a Fourier tensor product basis."""
    return TensorProductBasis2D(
        Fourier1D(n_modes_x, domain_x, mode=mode_x),
        Fourier1D(n_modes_y, domain_y, mode=mode_y)
    )


def create_mixed_basis_2d(basis_type_x: str, basis_type_y: str,
                          n_basis_x: int, n_basis_y: int,
                          domain_x: Tuple[float, float],
                          domain_y: Tuple[float, float],
                          **kwargs) -> TensorProductBasis2D:
    """Create a mixed tensor product basis.

    Parameters
    ----------
    basis_type_x, basis_type_y : str
        'chebyshev', 'cos', 'sin', or 'fourier' (cos+sin).
    """
    def make_1d_basis(basis_type, n_basis, domain):
        if basis_type == 'chebyshev':
            return Chebyshev1D(n_basis - 1, domain)
        elif basis_type == 'cos':
            return Fourier1D(n_basis, domain, mode='cos')
        elif basis_type == 'sin':
            return Fourier1D(n_basis, domain, mode='sin')
        elif basis_type == 'fourier':
            return Fourier1D(n_basis, domain, mode='both')
        else:
            raise ValueError(f"Unknown basis type: {basis_type}")

    return TensorProductBasis2D(
        make_1d_basis(basis_type_x, n_basis_x, domain_x),
        make_1d_basis(basis_type_y, n_basis_y, domain_y)
    )


def create_basis_2d(
    basis_type: str,
    N_x: int,
    N_y: int,
    x_domain: Tuple[float, float] = (0.0, 1.0),
    y_domain: Tuple[float, float] = (0.0, 1.0),
    elm_seed: int = 42,
    elm_activation: str = 'tanh',
) -> Union[TensorProductBasis2D, ELMBasis2D]:
    """Generic factory for 2D basis construction.

    Replaces the per-problem ``create_bratu_basis()``,
    ``create_burgers_basis()``, ``create_bl_basis()`` functions which
    were identical except for naming.

    Parameters
    ----------
    basis_type : str
        One of: 'chebyshev', 'sin_sin', 'cos_cos', 'fourier',
        'cheb_sin', 'sin_cheb', 'cheb_cos', 'cos_cheb',
        'cos_sin', 'sin_cos', 'elm'.
    N_x, N_y : int
        Basis size per direction. For Chebyshev, order = N-1.
        For Fourier sin/cos, n_modes = N.
        For ELM, total hidden units = N_x × N_y.
    x_domain, y_domain : tuple
        Physical domains.
    elm_seed : int
        Random seed for ELM (only used when basis_type='elm').
    elm_activation : str
        Activation for ELM ('tanh' or 'sigmoid').

    Returns
    -------
    TensorProductBasis2D or ELMBasis2D
    """
    bt = basis_type.lower().strip()

    if bt == 'chebyshev':
        return create_chebyshev_basis_2d(N_x - 1, N_y - 1, x_domain, y_domain)
    elif bt == 'sin_sin':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='sin', mode_y='sin')
    elif bt == 'cos_cos':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='cos', mode_y='cos')
    elif bt == 'fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='both', mode_y='both')
    elif bt == 'cheb_sin':
        return TensorProductBasis2D(Chebyshev1D(N_x - 1, x_domain), Fourier1D(N_y, y_domain, mode='sin'))
    elif bt == 'sin_cheb':
        return TensorProductBasis2D(Fourier1D(N_x, x_domain, mode='sin'), Chebyshev1D(N_y - 1, y_domain))
    elif bt == 'cheb_cos':
        return TensorProductBasis2D(Chebyshev1D(N_x - 1, x_domain), Fourier1D(N_y, y_domain, mode='cos'))
    elif bt == 'cos_cheb':
        return TensorProductBasis2D(Fourier1D(N_x, x_domain, mode='cos'), Chebyshev1D(N_y - 1, y_domain))
    elif bt == 'cos_sin':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='cos', mode_y='sin')
    elif bt == 'sin_cos':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='sin', mode_y='cos')
    elif bt == 'fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='both', mode_y='both')
    elif bt == 'cos_fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='cos', mode_y='both')
    elif bt == 'sin_fourier':
        return create_fourier_basis_2d(N_x, N_y, x_domain, y_domain, mode_x='sin', mode_y='both')
    elif bt == 'elm':
        return ELMBasis2D(
            n_hidden=N_x * N_y,
            domain_x=x_domain, domain_y=y_domain,
            activation=elm_activation, seed=elm_seed,
        )
    else:
        raise ValueError(
            f"Unknown basis_type '{bt}'. Supported: chebyshev, sin_sin, cos_cos, "
            f"fourier, cheb_sin, sin_cheb, cheb_cos, cos_cheb, cos_sin, sin_cos, elm"
        )


# ==============================================================================
#                  TENSOR PRODUCT BASIS (N-D)
# ==============================================================================

class TensorProductBasisND:
    """N-dimensional tensor product of 1D bases.

    Generalizes TensorProductBasis2D to arbitrary dimensions.
    Used for 3D+time problems like Beltrami flow.

    Parameters
    ----------
    bases : list of Basis1D
        One basis per dimension (e.g., [basis_x, basis_y, basis_z, basis_t]).
    """

    def __init__(self, bases):
        self.bases = list(bases)
        self.ndim = len(bases)
        self._sizes = [b.n_basis for b in bases]
        self.n_basis = int(np.prod(self._sizes))

    def evaluate(self, *coords):
        """Evaluate all basis functions at given coordinates.

        Parameters
        ----------
        *coords : arrays of shape (n_points,), one per dimension.

        Returns
        -------
        Phi : array of shape (n_points, n_basis)
        """
        return self._eval_deriv(coords, [0] * self.ndim)

    def derivative(self, *coords, orders):
        """Evaluate derivatives of all basis functions.

        Parameters
        ----------
        *coords : arrays of shape (n_points,), one per dimension.
        orders : list of int
            Derivative order for each dimension.

        Returns
        -------
        dPhi : array of shape (n_points, n_basis)
        """
        return self._eval_deriv(coords, orders)

    def _eval_deriv(self, coords, orders):
        n = len(np.atleast_1d(coords[0]))
        B1d = []
        for d in range(self.ndim):
            xd = np.atleast_1d(np.asarray(coords[d], dtype=np.float64))
            if orders[d] == 0:
                B1d.append(self.bases[d].evaluate(xd))
            else:
                B1d.append(self.bases[d].derivative(xd, order=orders[d]))

        result = B1d[0]
        for d in range(1, self.ndim):
            result = result[:, :, np.newaxis] * B1d[d][:, np.newaxis, :]
            result = result.reshape(n, -1)
        return result

    def reconstruct(self, coeffs, *coords):
        """Reconstruct f(coords) = Phi @ coeffs."""
        return self.evaluate(*coords) @ coeffs

    def __repr__(self):
        parts = [f"  dim{i}: {b}" for i, b in enumerate(self.bases)]
        return f"TensorProductBasisND(n_basis={self.n_basis},\n" + "\n".join(parts) + ")"


def _make_1d_basis(basis_type, N, domain):
    """Create a single 1D basis by type string."""
    bt = basis_type.lower()
    if bt == 'chebyshev':
        return Chebyshev1D(N - 1, domain)
    elif bt == 'cos':
        return Fourier1D(N, domain, mode='cos')
    elif bt == 'sin':
        return Fourier1D(N, domain, mode='sin')
    elif bt in ('both', 'fourier'):
        return Fourier1D(N, domain, mode='both')
    else:
        raise ValueError(f"Unknown basis type: {bt}")


def create_basis_nd(basis_type, N, domains,
                    per_dim_types=None, per_dim_N=None):
    """Create an N-dimensional tensor product basis.

    Parameters
    ----------
    basis_type : str
        Default type for all dimensions.
    N : int
        Default basis size for all dimensions.
    domains : list of tuple
        Domain (a, b) for each dimension.
    per_dim_types : list of str, optional
        Override type per dimension.
    per_dim_N : list of int, optional
        Override N per dimension.
    """
    ndim = len(domains)
    types = per_dim_types or [basis_type] * ndim
    sizes = per_dim_N or [N] * ndim
    bases = [_make_1d_basis(t, n, d) for t, n, d in zip(types, sizes, domains)]
    return TensorProductBasisND(bases)

