import numpy as np

class SheetTracker:
    def __init__(self, nbrs_in_mean=1):
        self.nbrs_in_mean = nbrs_in_mean
        self.mean = None

    def update(self, pts):
        pts = np.array([pts])
        if self.mean is None:
            self.mean = pts
        else:
            self.mean = np.concatenate((self.mean, pts))
            while len(self.mean) > self.nbrs_in_mean:
                self.mean = np.delete(self.mean, 0, 0)

    def get_mean(self):
        return np.mean(self.mean, axis=0)
