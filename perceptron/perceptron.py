import numpy as np


class perceptron(object):
    def __init__(self, learning_rate, data):
        self.learning_rate = learning_rate
        self.w1 = np.random.randn()
        self.w2 = np.random.randn()
        self.w3 = np.random.randn()
        self.b = np.random.randn()
        self.data = data

    def calcinput(self, data):
        return data[0] * self.w1 + data[1] * self.w2 + self.b

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_d(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def cost(self, prediction, target):
        return np.square(prediction - target)

    def cost_d(self, prediction, target):
        return 2 * (prediction - target)

    def cost_slope(self, derivativeCost, derivativePrediction):
        return derivativeCost * derivativePrediction

    def train(self, times):
        for i in range(times):
            ri = np.random.randint(len(self.data))
            point = self.data[ri]

            z = self.calcinput(point)

            prediction = self.sigmoid(z)
            target = point[2]

            # Cost Function
            cost = self.cost(prediction, target)

            # Derivative
            derivativePrediction = self.sigmoid_d(z)
            derivativeCost = self.cost_d(prediction, target)

            slopeCost = self.cost_slope(derivativeCost, derivativePrediction)

            derivativeW1 = point[0]
            derivativeW2 = point[1]
            derivativeB = 1

            dercost_derw1 = slopeCost * derivativeW1
            dercost_derw2 = slopeCost * derivativeW2
            dercost_derb = slopeCost * derivativeB

            # New Weight & Bias
            self.w1 = self.w1 - self.learning_rate * dercost_derw1
            self.w2 = self.w2 - self.learning_rate * dercost_derw2
            self.b = self.b - self.learning_rate * dercost_derb

            if i % 100 == 0:
                print("time try : ", i)
                print("data ke-", ri)
                print(point)
                print(prediction)

    def dataOut(self):
        for i in range(len(self.data)):
            point = self.data[i]
            print(point)

            z = self.calcinput(point)

            prediction = self.sigmoid(z)

            print("Prediction : %s" % format(prediction))

    def predict(self, dataPredict):
        z = self.calcinput(dataPredict)
        prediction = self.sigmoid(z)

        print(dataPredict)
        print("Prediction : %s" % format(prediction))
