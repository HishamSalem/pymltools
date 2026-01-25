"""
Cyclical Optimal Binning for Credit Risk
=========================================

Scikit-learn compatible transformer for IV-maximizing cyclical binning.

Usage:
    from cyclical_binner import CyclicalBinner
    
    binner = CyclicalBinner(gamma=0.02, k_max=6)
    binner.fit(temporal_values, targets)
    binned = binner.transform(temporal_values)
"""

import numpy as np
from numba import njit
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import time


# =============================================================================
# NUMBA CORE FUNCTIONS
# =============================================================================

@njit
def _circular_range_sum(prefix: np.ndarray, start: int, end: int, m: int) -> int:
    """Compute sum over circular range [start, end) using prefix sums."""
    if start < end:
        return prefix[end] - prefix[start]
    else:
        return (prefix[m] - prefix[start]) + prefix[end]


@njit
def _next_combination(combo: np.ndarray, n: int) -> bool:
    """Generate next combination in-place. Returns False when exhausted."""
    k = len(combo)
    i = k - 1
    while i >= 0 and combo[i] == n - k + i:
        i -= 1
    if i < 0:
        return False
    combo[i] += 1
    for j in range(i + 1, k):
        combo[j] = combo[j - 1] + 1
    return True


@njit
def _solve_for_k(
    k: int, m: int,
    prefix_events: np.ndarray, prefix_non_events: np.ndarray,
    total_events: int, total_non_events: int, total_samples: int,
    alpha_min: float, e_min: int, ne_min: int,
    lam: float, gamma: float
) -> tuple:
    """Enumerate all k-partitions and return best."""
    combo = np.arange(k, dtype=np.int64)
    total_partitions = 0
    valid_partitions = 0
    best_iv_reg = -np.inf
    best_splits = np.zeros(k, dtype=np.int64)
    min_bin_size = alpha_min * total_samples
    denom_p = total_events + k * lam
    denom_q = total_non_events + k * lam
    
    while True:
        total_partitions += 1
        is_valid = True
        iv = 0.0
        
        for j in range(k):
            start = combo[j]
            end = combo[(j + 1) % k]
            
            if start < end:
                e_count = prefix_events[end] - prefix_events[start]
                ne_count = prefix_non_events[end] - prefix_non_events[start]
            else:
                e_count = (prefix_events[m] - prefix_events[start]) + prefix_events[end]
                ne_count = (prefix_non_events[m] - prefix_non_events[start]) + prefix_non_events[end]
            
            if e_count < e_min or ne_count < ne_min or e_count + ne_count < min_bin_size:
                is_valid = False
                break
            
            p_j = (e_count + lam) / denom_p
            q_j = (ne_count + lam) / denom_q
            iv += (p_j - q_j) * np.log(p_j / q_j)
        
        if is_valid:
            valid_partitions += 1
            iv_reg = iv - gamma * k
            if iv_reg > best_iv_reg:
                best_iv_reg = iv_reg
                for i in range(k):
                    best_splits[i] = combo[i]
        
        if not _next_combination(combo, m):
            break
    
    return total_partitions, valid_partitions, best_iv_reg, best_splits


@njit
def _solve_core(
    m: int, prefix_events: np.ndarray, prefix_non_events: np.ndarray,
    total_events: int, total_non_events: int, total_samples: int,
    alpha_min: float, e_min: int, ne_min: int,
    lam: float, gamma: float, k_min: int, k_max: int
) -> tuple:
    """Core solver across all k values."""
    total_partitions = 0
    valid_partitions = 0
    best_iv_reg = -np.inf
    best_splits = np.zeros(k_max, dtype=np.int64)
    best_k = 0
    
    for k in range(k_min, k_max + 1):
        k_total, k_valid, k_best_iv, k_best_splits = _solve_for_k(
            k, m, prefix_events, prefix_non_events,
            total_events, total_non_events, total_samples,
            alpha_min, e_min, ne_min, lam, gamma
        )
        total_partitions += k_total
        valid_partitions += k_valid
        
        if k_best_iv > best_iv_reg:
            best_iv_reg = k_best_iv
            best_k = k
            for i in range(k):
                best_splits[i] = k_best_splits[i]
    
    return total_partitions, valid_partitions, best_iv_reg, best_splits, best_k


@njit
def _assign_bins(values: np.ndarray, splits: np.ndarray, m: int) -> np.ndarray:
    """Assign temporal values to bin indices."""
    n = len(values)
    k = len(splits)
    result = np.zeros(n, dtype=np.int64)
    
    for i in range(n):
        v = values[i]
        for j in range(k):
            start = splits[j]
            end = splits[(j + 1) % k]
            
            if start < end:
                if start <= v < end:
                    result[i] = j
                    break
            else:
                if v >= start or v < end:
                    result[i] = j
                    break
    
    return result


# =============================================================================
# RESULT DATACLASS
# =============================================================================

@dataclass
class BinningResult:
    """Container for binning results."""
    split_points: np.ndarray
    k: int
    iv_raw: float
    iv_smoothed: float
    iv_regularized: float
    woe: np.ndarray
    event_counts: np.ndarray
    non_event_counts: np.ndarray
    event_rates: np.ndarray
    partitions_evaluated: int
    valid_partitions: int
    solve_time_ms: float
    m: int
    
    def summary(self):
        """
        Return summary as pandas DataFrame (like OptBinning).
        
        Returns
        -------
        pd.DataFrame
            Binning table with columns: bin, range, count, count_%, 
            events, event_rate, woe, iv
        """
        import pandas as pd
        
        total_events = int(np.sum(self.event_counts))
        total_non_events = int(np.sum(self.non_event_counts))
        total_count = total_events + total_non_events
        
        rows = []
        for j in range(self.k):
            start = self.split_points[j]
            end = self.split_points[(j + 1) % self.k]
            rng = f"[{start}, {end})" if start < end else f"[{start}, {end})*"
            count = int(self.event_counts[j] + self.non_event_counts[j])
            
            p_j = self.event_counts[j] / total_events
            q_j = self.non_event_counts[j] / total_non_events
            iv_j = (p_j - q_j) * self.woe[j]
            
            rows.append({
                'bin': j,
                'range': rng,
                'count': count,
                'count_%': count / total_count * 100,
                'events': int(self.event_counts[j]),
                'non_events': int(self.non_event_counts[j]),
                'event_rate': float(self.event_rates[j]),
                'woe': float(self.woe[j]),
                'iv': float(iv_j)
            })
        
        df = pd.DataFrame(rows)
        
        # Add totals row
        totals_dict = {
            'bin': 'Totals',
            'range': '',
            'count': total_count,
            'count_%': 100.0,
            'events': total_events,
            'non_events': total_non_events,
            'event_rate': total_events / total_count,
            'woe': np.nan,
            'iv': self.iv_raw
        }
        df.loc[len(df)] = totals_dict
        
        return df
    
    def _interpret_iv(self) -> str:
        if self.iv_raw < 0.02:
            return "Not useful"
        elif self.iv_raw < 0.1:
            return "Weak"
        elif self.iv_raw < 0.3:
            return "Medium"
        elif self.iv_raw < 0.5:
            return "Strong"
        else:
            return "Suspicious (overfit?)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'split_points': self.split_points.tolist(),
            'k': self.k,
            'iv_raw': self.iv_raw,
            'iv_smoothed': self.iv_smoothed,
            'iv_regularized': self.iv_regularized,
            'woe': self.woe.tolist(),
            'event_counts': self.event_counts.tolist(),
            'non_event_counts': self.non_event_counts.tolist(),
            'event_rates': self.event_rates.tolist(),
        }
    
    def woe_table(self) -> List[Dict]:
        """WOE table for scorecard documentation."""
        return [
            {
                'bin': j,
                'start': int(self.split_points[j]),
                'end': int(self.split_points[(j + 1) % self.k]),
                'events': int(self.event_counts[j]),
                'non_events': int(self.non_event_counts[j]),
                'event_rate': float(self.event_rates[j]),
                'woe': float(self.woe[j])
            }
            for j in range(self.k)
        ]


# =============================================================================
# SKLEARN-COMPATIBLE TRANSFORMER
# =============================================================================

class CyclicalBinner(BaseEstimator, TransformerMixin):
    """
    Cyclical optimal binning transformer.
    
    Finds the IV-maximizing partition of a cyclical temporal feature
    into contiguous circular bins, subject to statistical constraints.
    
    Parameters
    ----------
    m : int, default=24
        Cardinality of temporal domain (e.g., 24 for hours, 12 for months).
        
    gamma : float, default=0.02
        Regularization penalty per bin. Higher values prefer fewer bins.
        
    lam : float, default=0.5
        Laplace smoothing parameter for WOE stability.
        
    alpha_min : float, default=0.05
        Minimum fraction of total samples per bin.
        
    e_min : int, default=10
        Minimum events per bin.
        
    ne_min : int, default=10
        Minimum non-events per bin.
        
    k_min : int, default=2
        Minimum number of bins.
        
    k_max : int, default=6
        Maximum number of bins.
    
    Attributes
    ----------
    result_ : BinningResult
        Complete binning result after fitting.
        
    split_points_ : np.ndarray
        Optimal split points.
        
    n_bins_ : int
        Number of bins in optimal solution.
        
    woe_ : np.ndarray
        Weight of Evidence for each bin.
        
    iv_ : float
        Information Value (raw, unsmoothed).
    
    Examples
    --------
    >>> binner = CyclicalBinner(m=24, gamma=0.02)
    >>> binner.fit(hours, targets)
    >>> binned = binner.transform(hours)
    >>> print(binner.result_.summary())
    """
    
    def __init__(
        self,
        m: int = 24,
        gamma: float = 0.02,
        lam: float = 0.5,
        alpha_min: float = 0.05,
        e_min: int = 10,
        ne_min: int = 10,
        k_min: int = 2,
        k_max: int = 6
    ):
        self.m = m
        self.gamma = gamma
        self.lam = lam
        self.alpha_min = alpha_min
        self.e_min = e_min
        self.ne_min = ne_min
        self.k_min = k_min
        self.k_max = k_max
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the optimal binning.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Temporal values (0 to m-1).
            
        y : array-like of shape (n_samples,)
            Binary target (1=event, 0=non-event).
        
        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=np.int64).ravel()
        y = np.asarray(y, dtype=np.int64).ravel()
        
        if len(X) != len(y):
            raise ValueError(f"X and y must have same length: {len(X)} vs {len(y)}")
        if not np.all((X >= 0) & (X < self.m)):
            raise ValueError(f"X values must be in [0, {self.m - 1}]")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("y must be binary (0 or 1)")
        
        # Aggregate
        events = np.bincount(X[y == 1], minlength=self.m).astype(np.int64)
        non_events = np.bincount(X[y == 0], minlength=self.m).astype(np.int64)
        
        total_events = int(np.sum(events))
        total_non_events = int(np.sum(non_events))
        total_samples = total_events + total_non_events
        
        # Prefix sums
        prefix_events = np.zeros(self.m + 1, dtype=np.int64)
        prefix_non_events = np.zeros(self.m + 1, dtype=np.int64)
        prefix_events[1:] = np.cumsum(events)
        prefix_non_events[1:] = np.cumsum(non_events)
        
        # Solve
        start_time = time.perf_counter()
        
        total_partitions, valid_partitions, best_iv_reg, best_splits, best_k = _solve_core(
            self.m, prefix_events, prefix_non_events,
            total_events, total_non_events, total_samples,
            self.alpha_min, self.e_min, self.ne_min,
            self.lam, self.gamma, self.k_min, self.k_max
        )
        
        solve_time = (time.perf_counter() - start_time) * 1000
        
        if best_k == 0:
            raise ValueError(
                "No valid partition found. Try relaxing constraints: "
                f"alpha_min={self.alpha_min}, e_min={self.e_min}, ne_min={self.ne_min}"
            )
        
        # Store fitted attributes
        self.split_points_ = best_splits[:best_k].copy()
        self.n_bins_ = best_k
        
        # Compute metrics
        event_counts = np.zeros(best_k, dtype=np.int64)
        non_event_counts = np.zeros(best_k, dtype=np.int64)
        
        for j in range(best_k):
            start = self.split_points_[j]
            end = self.split_points_[(j + 1) % best_k]
            event_counts[j] = _circular_range_sum(prefix_events, start, end, self.m)
            non_event_counts[j] = _circular_range_sum(prefix_non_events, start, end, self.m)
        
        # WOE and IV
        woe = np.zeros(best_k, dtype=np.float64)
        iv_smooth = 0.0
        denom_p = total_events + best_k * self.lam
        denom_q = total_non_events + best_k * self.lam
        
        for j in range(best_k):
            p_j = (event_counts[j] + self.lam) / denom_p
            q_j = (non_event_counts[j] + self.lam) / denom_q
            woe[j] = np.log(p_j / q_j)
            iv_smooth += (p_j - q_j) * woe[j]
        
        iv_raw = 0.0
        for j in range(best_k):
            p_j = event_counts[j] / total_events
            q_j = non_event_counts[j] / total_non_events
            if p_j > 0 and q_j > 0:
                iv_raw += (p_j - q_j) * np.log(p_j / q_j)
        
        self.woe_ = woe
        self.iv_ = iv_raw
        
        event_rates = event_counts / (event_counts + non_event_counts)
        
        self.result_ = BinningResult(
            split_points=self.split_points_,
            k=best_k,
            iv_raw=iv_raw,
            iv_smoothed=iv_smooth,
            iv_regularized=best_iv_reg,
            woe=woe,
            event_counts=event_counts,
            non_event_counts=non_event_counts,
            event_rates=event_rates,
            partitions_evaluated=total_partitions,
            valid_partitions=valid_partitions,
            solve_time_ms=solve_time,
            m=self.m
        )
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform temporal values to bin indices.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Temporal values (0 to m-1).
        
        Returns
        -------
        X_binned : ndarray of shape (n_samples,)
            Bin index for each value (0 to n_bins_-1).
        """
        check_is_fitted(self, ['split_points_', 'n_bins_'])
        X = np.asarray(X, dtype=np.int64).ravel()
        return _assign_bins(X, self.split_points_, self.m)
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
    
    def get_woe_encoder(self) -> Dict[int, float]:
        """
        Get WOE encoding dictionary.
        
        Returns
        -------
        dict
            Mapping from bin index to WOE value.
        """
        check_is_fitted(self, ['woe_', 'n_bins_'])
        return {i: float(self.woe_[i]) for i in range(self.n_bins_)}
    
    def transform_woe(self, X: np.ndarray) -> np.ndarray:
        """
        Transform temporal values directly to WOE values.
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Temporal values.
        
        Returns
        -------
        X_woe : ndarray of shape (n_samples,)
            WOE value for each sample.
        """
        check_is_fitted(self, ['woe_'])
        bins = self.transform(X)
        return self.woe_[bins]

