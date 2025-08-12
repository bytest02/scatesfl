
class PerfCollector():
    def __init__(self, config) -> None:
        self.config = config
        self.repo = []

    def collect(self, data) -> None:
        self.repo.append(data)