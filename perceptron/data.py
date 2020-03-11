def dataVehicle():
    # each point is mass, Length, Class (0.1)
    # 0 for Lorry and 1 for Van
    data_vehicle = [[10.0, 6, 0],
                    [20.0, 5, 0],
                    [5.0, 4, 1],
                    [3.0, 3, 1],
                    [2.0, 5, 1],
                    [2.0, 5, 1],
                    [3.0, 6, 0],
                    [10.0, 7, 0],
                    [15.0, 8, 0],
                    [5.0, 9, 0]]

    mystery_vehicle = [4.0, 2]

    return data_vehicle, mystery_vehicle


def dataFlower():
    # each point is Sepal length on Cm, Sepal width on Cm & type (0.1)
    # 0 for Iris-setosa & 1 for Iris-versicolor
    data = [[5.1,  3.5,  0],
            [4.9,  3.,  0],
            [4.7,  3.2,  0],
            [4.6,  3.1,  0],
            [5.,  3.6,  0],
            [5.4,  3.9,  0],
            [4.6,  3.4,  0],
            [5.,  3.4,  0],
            [4.4,  2.9,  0],
            [4.9,  3.1,  0],
            [5.4,  3.7,  0],
            [4.8,  3.4,  0],
            [4.8,  3.,  0],
            [4.3,  3.,  0],
            [5.8,  4.,  0],
            [5.7,  4.4,  0],
            [5.4,  3.9,  0],
            [5.1,  3.5,  0],
            [5.7,  3.8,  0],
            [5.1,  3.8,  0],
            [7.,  3.2,  1],
            [6.4,  3.2,  1],
            [6.9,  3.1,  1],
            [5.5,  2.3,  1],
            [6.5,  2.8,  1],
            [5.7,  2.8,  1],
            [6.3,  3.3,  1],
            [4.9,  2.4,  1],
            [6.6,  2.9,  1],
            [5.2,  2.7,  1],
            [5.,  2.,  1],
            [5.9,  3.,  1],
            [6.,  2.2,  1],
            [6.1,  2.9,  1],
            [5.6,  2.9,  1],
            [6.7,  3.1,  1],
            [5.6,  3.,  1],
            [5.8,  2.7,  1],
            [6.2,  2.2,  1],
            [5.6,  2.5,  1],
            [5.9,  3.2,  1],
            [6.1,  2.8,  1],
            [6.3,  2.5,  1],
            [6.1,  2.8,  1],
            [6.4,  2.9,  1]]
    mystery_flower = [5.1, 3.7]

    return data, mystery_flower