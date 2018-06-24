# --------------------------------------------------------------------------
# ------------  Metody Systemowe i Decyzyjne w Informatyce  ----------------
# --------------------------------------------------------------------------
#  Zadanie 4: Zadanie zaliczeniowe
#  autorzy: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np

from network import Network, load


def predict(test_data):
    """
    Funkcja pobiera macierz przykladow zapisanych w macierzy X o wymiarach NxD i zwraca wektor y o wymiarach Nx1,
    gdzie kazdy element jest z zakresu {0, ..., 35} i oznacza znak rozpoznany na danym przykladzie.
    :param test_data: macierz o wymiarach NxD
    :return: wektor o wymiarach Nx1
    """
    krok = 1

    bias, weight = load('net.pkl')
    net = Network([int(3136 / krok), 30, 40, 36], bias, weight)

    data_x = []

    for main_data in test_data:
        data_x.append(main_data[::krok])

    x = []

    for dx in data_x:
        x.append(np.array(dx).reshape((len(dx), 1)))

    values = net.predict(x)
    print(values)
    values = np.reshape(values, (len(values), 1))
    print(np.shape(values))
    return values
