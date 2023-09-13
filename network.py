#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np
"""
Hacemos uso (importamos) de random y numpy, que son librerias que contienen funciones (codigo que ya ha sido creado por
otras personas) que nos seran de utilidad a lo largo del presente codigo.
"""
class Network(object):
    """
    Se creo la clase Network, la cual contiene los atributos:
    -num_layers, el cual sera el numero de capas que contiene la red neuronal
    -sizes, que es una lista con el numero de neuronas en la respectiva capa de la red neuronal, por ejemlo, si la lista
    fuera [2, 3, 1] entonces seria una red neuronal de 3 capas con 2 neuronas en la primera capa, 3 neuronas en la
    segunda capa y 1 neurona en la capa final.
    -'Biases' (sesgos) y 'Weights' (pesos).
    """
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

        """Los biases y weights son inicializados de forma aleatoria a traves de la funcion np.random.randn,los biases 
        inicia en '1' ya que la capa '0' es la capa de entradas y en esta se omite establecer cualquier sesgo para esas 
        neuronas, ya que los sesgos solo se utilizan para calcular las salidas de capas posteriores. 
        """

    def feedforward(self, a):
        """En este metodo, dados los biases y weights hacemos uso de la funcion sigmoide (funcion de activacion)
        para la activacion de las neuronas."""
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a


    def SGD_momentum(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, Beta=0.1):
        """Stochastic gradient descent (SGD) es un método que calcula las predicciones y errores de 1 elemento escogido
         aleatoriamente de nuestro minibatch.  Los datos de entrenamiento (`training_data`) es una lista de datos que
         "capacitaran" a la red neuronal hasta que aprenda cómo proporcionar la respuesta adecuada.
         Los datos de prueba (test data) son son los datos que nos 'reservamos' para comprobar si el modelo que hemos
         generado a partir de los datos de entrenamiento 'funciona'. Despues de cada epoca se comprobara la eficacia
         de la red."""

        training_data = list(training_data)
        n = len(training_data)
        """la funcion len(training_data) nos da el numero de elementos de los datos de entrenamiento."""
        if test_data:
            test_data = list(test_data)
            n_test = len(test_data)


        """Las variables epochs y mini_batch_size son la cantidad de épocas para entrenar y el tamaño de los 
        minibatches respectivamente, que se usarán al realizar el muestreo. Eta 'η' es la tasa de aprendizaje. 
        El programa evaluará la eficacia de la red después de cada época de entrenamiento e 'imprimira' el progreso 
        parcial. Esto nos indicara el progreso y que tan eficaz es la red, otra forma de hacerlo pudo haber sido la 
        elaboracion de una grafica de 'loss' vs epocas."""
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {} : {} / {}".format(j,self.evaluate(test_data),n_test))
            else:
                print("Epoch {} complete".format(j))

    def update_mini_batch(self, mini_batch, eta, Beta=0.1):
        """La red mejorara haciendo que la funcion de costo sea menor, esto a traves de ir modificando los pesos y
         sesgos aplicando el SGD mediante Back propagation a un minibatch. Entonces nabla_b sera la aproximacion al
         gradiente de la funcion de costo respecto a cada sesgo y nabla_w la aproximacion al gradiente de la funcion de
         costo respecto a cada peso."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        v_t_previousw = [np.zeros(w.shape) for w in self.weights]
        v_t_previousb = [np.zeros(b.shape) for b in self.biases]
        v_wt = [np.zeros(w.shape) for w in self.weights]
        v_bt = [np.zeros(b.shape) for b in self.biases]


        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            v_wt = [Beta * v_wtp + (1 - Beta) * nw+dnw for v_wtp, nw, dnw in zip(v_t_previousw, nabla_w, delta_nabla_w)]
            v_bt = [Beta * v_wtp + (1 - Beta) * nb+dnb for v_wtp, nb, dnb in zip(v_t_previousb, nabla_b, delta_nabla_b)]

            v_t_previousw = v_wt
            v_t_previousb = v_bt

        self.weights = [w-(eta/len(mini_batch))*vw
                        for w, vw in zip(self.weights, v_wt)]
        self.biases = [b-(eta/len(mini_batch))*vb
                       for b, vb in zip(self.biases, v_bt)]

    def backprop(self, x, y):
        """(nabla_b, nabla_w) representan el gradiente de la función de costo C_x. ``nabla_b``
         y ``nabla_w`` son listas capa por capa de matrices numpy."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        activations_soft = [] # capa softmax
        for z in (zs):
            activation_soft = softmax(z)
            activations_soft.append(activation_soft)




        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
"""Importamos el archivo mnist_loader, el cual arroja los datos de la base de datos MNIST como una tupla que contiene
los datos de entrenamiento, los datos de validacion y los datos de prueba.
."""
net = Network([784, 30, 10])
"""Para esta red, pondremos una capa inicial de 784 neuronas, 30 neuronas en la capa intermedia y 10 neuronas en la capa
final (1 neurona por cada digito).
"""
net.SGD_momentum(training_data, 30, 10, 3.0, test_data=test_data)
"""La red tendra 30 epocas, minibatches de tamaño 10, una tasa de aprendizaje de 3.0 y los datos de prueba se tomaran
del archivo mnist_loader que hemos importado.
"""
