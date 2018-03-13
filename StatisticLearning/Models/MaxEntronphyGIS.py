import numpy as np

class MaxEntronPhyGIS:
    # iternate times
    iter = None
    def __init__(self, iter=1000):
        self.iter = iter

    def train(self, x, y):
