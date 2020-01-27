from abc import ABC, abstractmethod


class BannerReplacer(ABC):
    @abstractmethod
    def build_model(self):
        pass

    @abstractmethod
    def detect_banner(self):
        pass

    @abstractmethod
    def insert_banner(self):
        pass
