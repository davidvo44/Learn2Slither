class Parameter:
    def __init__(self):
        self.learningRate = 0.8
        self.discountFactor = 0.95
        self.epoch = 1000
        self.explorationProb = 0.2

    def __setattr__(self, name, value):
        self.__dict__[name] = value

    def __getattr__(self, item):
        return (self.__dict__[item])
    