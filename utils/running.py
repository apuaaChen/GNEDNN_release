class average:
    def __init__(self):
        self.value = 0
        self.counter = 0

    def add(self, v):
        self.value = float(self.value * self.counter + v)/float(self.counter + 1)
        self.counter += 1

    def clear(self):
        self.value = 0
        self.counter = 0

    def print(self):
        print(self.value)