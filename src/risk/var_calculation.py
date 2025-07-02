"""
Value at Risk (VaR) calculation for trading system.
"""
import numpy as np
from typing import Any

def calculate_var(returns: np.ndarray, confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at a given confidence level.
    """
    return np.percentile(returns, (1-confidence)*100) 