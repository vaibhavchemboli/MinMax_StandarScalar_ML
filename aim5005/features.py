import numpy as np
from typing import List, Tuple
### YOU MANY NOT ADD ANY MORE IMPORTS (you may add more typing imports)

class MinMaxScaler:
    def __init__(self):
        self.minimum = None
        self.maximum = None
        
    def _check_is_array(self, x: np.ndarray) -> np.ndarray:
        if not isinstance(x, np.ndarray):
            x = np.array(x)
            
        assert isinstance(x, np.ndarray), "Expected the input to be a NumPy array"
        return x
        
    def fit(self, x: np.ndarray) -> None:   
        x = self._check_is_array(x)
        self.minimum = x.min(axis=0)
        self.maximum = x.max(axis=0)
        
    def transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        diff_max_min = self.maximum - self.minimum
        
        # Corrected calculation
        return (x - self.minimum) / diff_max_min
    
    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        x = self._check_is_array(x)
        self.fit(x)
        return self.transform(x)
    
    
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std_dev = None

    def fit(self, x: np.ndarray) -> None:
        self.mean = np.mean(x, axis=0)
        self.std_dev = np.std(x, axis=0)

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std_dev

    def fit_transform(self, x: np.ndarray) -> np.ndarray:
        self.fit(x)
        return self.transform(x)
    
"""References:

StackOverflow, 2017, https://stackoverflow.com/questions/40758562/can-anyone-explain-me-standardscaler

Kenzo, 2016, https://kenzotakahashi.github.io/scikit-learns-useful-tools-from-scratch.html

"""