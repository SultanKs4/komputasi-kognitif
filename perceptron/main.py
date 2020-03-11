from data import dataFlower
from perceptron import perceptron

if __name__ == "__main__":
    data = dataFlower()
    percep = perceptron(0.8, data[0])
    percep.train(10000)
    percep.dataOut()
    percep.predict(data[1])
