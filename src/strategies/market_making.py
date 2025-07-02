"""
Market Making trading strategy.
"""
import numpy as np
from typing import Any

class MarketMakingStrategy:
    """
    Market making strategy for trading.
    """
    def __init__(self, spread: float = 0.01):
        self.spread = spread

    def generate_quotes(self, price: float) -> dict:
        """Generate bid and ask quotes."""
        return {'bid': price - self.spread/2, 'ask': price + self.spread/2} 