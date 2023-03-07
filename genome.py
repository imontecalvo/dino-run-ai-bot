import numpy as np


class Genome():
    def __init__(self, architecture, genomes=[]):
        all_comb = np.prod(np.array(architecture))//architecture[-1]
        self.architecture = architecture
        self.activation = Genome.relu
        if len(genomes) == 0:
            self.weights, self.biases = Genome.random_params(architecture)
        else:
            self.weights, self.biases = Genome.mix_genomes(genomes)

    # Funciones de activacion
    def relu(x):
        return np.maximum(0, x)

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # Asignacion de pesos
    def mix_genomes(genomes):
        weights = {}
        biases = {}

        # Asigno importancia a cada gen para un promedio ponderado
        avg_w = []
        for i in range(len(genomes)):
            prev = len(genomes) if i == 0 else avg_w[i-1]
            avg_w.append(prev/2+0.5)

        # Recorro capas (pesos) -> w1,w2,...,wn
        for i in range(1, len(genomes[0].architecture)):
            # Promedio entre los mismos pesos de cada red neuronal (genoma)
            # w1G1=[[a,b],[c,d]] ; w1G2=[[e,f],[g,h]] ; w1G3=[[i,j],[k,l]]
            # =>  w1G4 = [ [mean(a,e,i) , mean(b,f,j)] , [mean(c,g,j) , mean(d,h,l)] ]
            current_weights = [genome.get_weight(i) for genome in genomes]
            # avg_weight = np.sum(current_weights, axis=0)/len(current_weights)
            avg_weight = np.average(current_weights, axis=0, weights=avg_w)
            weights["w"+str(i)] = Genome.apply_mutation(avg_weight)

            current_biases = [genome.get_bias(i) for genome in genomes]
            # avg_bias = np.sum(current_biases,axis=0)/len(current_biases)
            avg_bias = np.average(current_biases, axis=0, weights=avg_w)
            biases["b"+str(i)] = Genome.apply_mutation(avg_bias)

        return weights, biases

    def apply_mutation(x):
        x_size = np.size(x)
        # Mutamos el entre el 0 y 30% de los valores
        max_mutations = x_size*0.3
        n_mutations = np.random.randint(low=0, high= max_mutations if max_mutations>=1 else 1 )
        # Seleccionamos aleatoriamente el indice de los valores a mutar
        for idx in np.random.randint(low=0, high=x_size-1, size=n_mutations):
            value = np.random.rand()*10-5
            x.flat[idx] = value
        return x

    def random_params(architecture):
        weights = {}
        biases = {}
        for i in range(1, len(architecture)):
            weights["w"+str(i)] = (np.random.rand(architecture[i],
                                                  architecture[i-1]))*10-5
            biases["b"+str(i)] = (np.random.rand(architecture[i], 1))*10-5
        return weights, biases

    # Obtener pesos y bias
    def get_weight(self, i):
        return self.weights["w"+str(i)]

    def get_bias(self, i):
        return self.biases["b"+str(i)]

    # Obtener output
    def evaluate(self, data):
        self.results = {}
        self.results["y0"] = np.array(data).T

        for i in range(1, len(self.architecture)):
            self.results["z"+str(i)] = self.weights["w"+str(i)
                                                    ]@self.results["y"+str(i-1)]+self.biases["b"+str(i)]
            if i < len(self.architecture)-1:
                self.results["y" +
                             str(i)] = self.activation(self.results["z"+str(i)])
            else:
                self.results["y" +
                             str(i)] = self.softmax(self.results["z"+str(i)])

        output = self.results["y"+str(len(self.architecture)-1)]
        return output, np.argmax(output)
