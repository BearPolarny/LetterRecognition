import pickle
import time

import numpy as np

from network import Network, load

main_data = pickle.load(open('train.pkl', mode='rb'))

data_x_long = main_data[0]
data_y_bad = main_data[1]

data_x = []

krok = 2

for main_data in data_x_long:
    data_x.append(main_data[::krok])

data_y = np.zeros((len(data_y_bad), 36))
for main_data, label in zip(data_y, data_y_bad):
    main_data[label] = 1

x = []
y = []

for dx, dy in zip(data_x, data_y):
    x.append(np.array(dx).reshape((len(dx), 1)))
    y.append(np.array(dy).reshape((len(dy), 1)))

train_data_x = x[:-1000]
train_data_y = y[:-1000]

# train_data_x = train_data_x[:2500]
# train_data_y = train_data_y[:2500]


test_data_x = x[-1000:]
test_data_y = data_y_bad[-1000:]

test = list(zip(test_data_x, test_data_y))
train = list(zip(train_data_x, train_data_y))


def moja():
    bias, weight = load('net82.9_k2.pkl')

    net = Network([int(3136 / krok), 30, 40, 36], bias, weight)
    czas = time.localtime(time.time())
    strCzas = str(czas.tm_hour) + ':' + str(czas.tm_min) + ':' + str(czas.tm_sec)
    print('Witam o ' + strCzas, end='')
    if czas.tm_hour < 5:
        print(' Późno już... Może kawy?')
    elif czas.tm_hour < 10:
        print('Jest dość wcześnie. Nie śpisz już, czy jeszcze? Może kawy?')
    else:
        print()

    czas = time.time()

    net.stochastic_gradient_descent(train, 100, 10, 6.0, test)

    czas2 = time.time()
    print('Nauczone. Czas na egzammin')

    evaluation = net.evaluate(test[1000:])

    czas3 = time.time()

    print(str(evaluation) + ' out of ' + str(len(test[1000:])) + ', it\'s ' + str(
        100 * round(evaluation / len(test[1000:]), 2)) + '%' +
          '\n Training duration: ' + str(round(czas2 - czas, 2)) +
          '\n Eval duration: ' + str(round(czas3 - czas2, 2)))

    net.save()


if __name__ == '__main__':
    from predict import predict

    czas = time.time()

    # print(test_data_x[1])
    # print(data_x[1])
    # time.sleep(20)

    # v = predict(test_data_x)
    v = predict(data_x[2000:3000])
    s = 0
    # for v1, v2 in zip(v, test_data_y):
    for v1, v2 in zip(v, data_y_bad[2000:3000]):
        # print(str(v1) + ' ' + str(v2) + '\n')
        if v1 == v2:
            s += 1
    print("{} / {} it\'s {}% in {}s".format(s, len(v), round(100 * s / len(v), 2), round(time.time() - czas, 2)))
