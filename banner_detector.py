import cv2 as cv
import numpy as np
from abc import ABC, abstractmethod


class BannerInception(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def detect_field(self, frame):
        pass

    @abstractmethod
    def banner_imputation(self):
        pass
