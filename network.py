import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np


# from time import sleep as zzz


def sig(z):
    return 1.0 / (1.0 + np.exp(-z))


def sig_pr(z):
    return sig(z) * (1 - sig(z))


def cost_derivative(output_activations, y):
    return output_activations - y


def load(path='net.pkl'):
    try:
        biases, weights = pickle.load(open(path, mode='rb'))
    except FileNotFoundError:
        raise NameError('Nie znaleziono pliku')

    return biases, weights


class Network(object):

    def __init__(self, sizes, bias=None, weight=None):
        self.layers_count = len(sizes)
        self.sizes = sizes
        if bias is None:
            self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        else:
            self.biases = bias
            self.weights = weight

    def feedforward(self, x):

        for bias, weight in zip(self.biases, self.weights):
            x = sig(np.dot(weight, x) + bias)
        return x

    def stochastic_gradient_descent(self, training_data, epochs, mini_batch_size, eta, test_data=None):

        training_data = list(training_data)
        n = len(training_data)

        percenty = []

        if test_data is not None:
            test_data = list(test_data)
            test_len = len(test_data)
        else:
            test_len = 1

        best_percent = 0

        for i in range(epochs):

            t = time.time()
            random.shuffle(training_data)
            mini_batches = [training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)]

            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                estimate = self.evaluate(test_data)
                percent = 100 * estimate / test_len
                percenty.append(percent)
                if percent > best_percent and percent > 70:
                    best_percent = percent
                    self.save('net' + str(round(percent, 1)) + '_k2.pkl')
                print("Epoch {} : {}/{} its {}% w {}s".format(i, estimate, test_len, round(percent, 2),
                                                              round(time.time() - t, 2)))
            else:
                print("Epoch {} complete".format(i))

        plt.plot(percenty)
        plt.show()

    def update_mini_batch(self, mini_batch, eta):

        biases_n = [np.zeros(bias.shape) for bias in self.biases]
        weights_n = [np.zeros(weight.shape) for weight in self.weights]
        for x, y in mini_batch:
            delta_b, delta_w = self.backprop(x, y)
            biases_n = [bias_n + delta for bias_n, delta in zip(biases_n, delta_b)]
            weights_n = [weight_n + delta for weight_n, delta in zip(weights_n, delta_w)]
        self.weights = [weight - (eta / len(mini_batch)) * weight_n for weight, weight_n in
                        zip(self.weights, weights_n)]
        self.biases = [bias - (eta / len(mini_batch)) * bias_n for bias, bias_n in zip(self.biases, biases_n)]

    def backprop(self, x, y):

        bias_n = [np.zeros(bias.shape) for bias in self.biases]
        weight_n = [np.zeros(weight.shape) for weight in self.weights]

        layer_activation = x
        layers_activations = [x]
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = weight @ layer_activation + bias
            zs.append(z)
            layer_activation = sig(z)
            layers_activations.append(layer_activation)

        delta = cost_derivative(layers_activations[-1], y) * sig_pr(zs[-1])
        bias_n[-1] = delta
        weight_n[-1] = delta @ layers_activations[-2].T

        for layer in range(2, self.layers_count):
            z = zs[-layer]
            sig_prime = sig_pr(z)
            delta = self.weights[-layer + 1].T @ delta * sig_prime
            bias_n[-layer] = delta
            weight_n[-layer] = delta @ layers_activations[-layer - 1].T
        return bias_n, weight_n

    def evaluate(self, test_data):

        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def predict(self, data):

        test_results = []

        for x in data:
            a = self.feedforward(x)
            # print(a.shape)
            b = np.argmax(a)
            # print(b)
            # print('')
            test_results.append(b)

        return test_results

    def save(self, path='net.pkl'):
        pickle.dump((self.biases, self.weights), open(path, mode='wb'))
