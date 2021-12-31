# Fixed power generator
class generator():
    def __init__(self, capacity=20e3):
        self.capacity = capacity

    def generate_power(self):
        return self.capacity
