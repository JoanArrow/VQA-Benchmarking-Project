
"""
Regular Quantum Approximate Boltzmann Machine (QABoM) algorithm
as described in [ADD PAPER] ------------------------------------------------
"""


# Modules
import pennylane as qml
from pennylane import numpy as np
from pennylane import qaoa
import copy
import json
import time


class qRBM:
    """
    desc
    """

    def __init__(self, num_visible, num_hidden, device_name, beta_temp=2.0, optimizer_steps=70, bitFlipNoise=False):
        """
        Constructor for the qRBM
        :param num_visible:
        :param num_hidden:
        :param device_name:
        :param beta_temp:
        :param optimizer_steps:
        :param trackErrors:
        """
        # Setting up the neural net
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_total = num_visible + num_hidden               # NOTE: total does not include ancillaries
        self.num_ancillaries = self.num_total
        # For labelling & iteration purposes
        self.vis_indices = [i for i in range(num_visible)]
        self.hid_indices = [i + num_visible for i in range(num_hidden)]
        self.tot_indices = [i for i in range(self.num_total)]
        self.anc_indices = [i + self.num_total for i in range(self.num_total)]

        # Values for preparing the initial thermal state
        self.beta_temp = beta_temp
        self.state_prep_angle = np.arctan(np.exp(-beta_temp/2.0)) * 2.0

        # Other params
        # Using Pennylane's gradient descent optimizer
        self.optimizer = qml.GradientDescentOptimizer()
        self.optimizer_steps = optimizer_steps
        self.bitFlipNoise = bitFlipNoise
        self.stopping_criteria = 0.0016
        self.hellinger = []
        self.CF = []

        # Initializing the weights using a randomly generated uniform distribution
        weights_param = 0.1 * np.sqrt(6.0 / self.num_total)
        # If only 1 visible node and 1 hidden node, there's only 1 weight; initialize differently
        if num_visible == 1 and num_hidden == 1:
            self.WEIGHTS = np.random.uniform(low=-weights_param, high=weights_param)
        else:
            self.WEIGHTS = np.asarray(
                np.random.uniform(low=-weights_param, high=weights_param, size=(num_visible, num_hidden))
            )

        # Initializing the full mixer Hamiltonian
        coeffs_fHM = np.array([])
        obs_fHM = []
        for i in self.tot_indices:
            coeffs_fHM = np.append(coeffs_fHM, 1.0)
            obs_fHM.append(qml.PauliZ(i))
        self.fullMixerH = qml.Hamiltonian(coeffs_fHM, obs_fHM)

        # Initializing the full cost Hamiltonian
        self.updateHamiltonians()

        self.device = qml.device(device_name, wires=(self.num_total + self.num_ancillaries))


    def updateHamiltonians(self):
        """
        Updates the cost Hamiltonian based on the current weights
        :return:
        """
        # Generate the full cost Hamiltonian
        coeffs_fHC = np.array([])
        obs_fHC = []
        # Need to generate differently for minimal RBM
        if self.num_visible == 1 and self.num_hidden == 1:
            coeffs_fHC = np.append(coeffs_fHC, self.WEIGHTS)
            obs_fHC.append(qml.PauliZ(0) @ qml.PauliZ(1))
        # All other RBMs
        else:
            for i in self.vis_indices:
                for j in self.hid_indices:
                    coeffs_fHC = np.append(coeffs_fHC, self.WEIGHTS[i][j - self.num_visible])
                    obs_fHC.append(qml.PauliZ(i) @ qml.PauliZ(j))
        # Put it all together
        self.fullCostH = qml.Hamiltonian(coeffs_fHC, obs_fHC)


    def unclamped_qaoa_layer(self, gamma, nu):
        """
        One layer of the QAOA circuit used in the unclamped sampling procedure
        :param gamma:
        :param nu:
        :return:
        """
        # Using Pennylane's built-in QAOA module for this
        qaoa.cost_layer(gamma, self.fullCostH)
        qaoa.mixer_layer(nu, self.fullMixerH)


    def unclamped_qaoa_circuit(self, params):
        """
        Initializes and layers the QAOA circuit used in the unclamped sampling procedure
        :param params:
        :return:
        """
        # Initialize the circuit
        for node in self.tot_indices:
            # Get the corresponding ancillary qubit
            anc = node + self.num_total
            # RX on ancillaries using angle prepared in initialization
            qml.RX(phi=self.state_prep_angle, wires=anc)
            # Entangle ancillaries with main qubits
            qml.CNOT([anc, node])

        # Layer the QAOA circuit
        # Currently using depth=2                                                                      REVIEW??????????
        qml.layer(self.unclamped_qaoa_layer, depth=2, gamma=params[0], nu=params[1])


    def unclamped_sampling(self, params):
        """
        The unclamped sampling procedure as described in the paper
        :param params:
        :return:
        """
        # The cost function needed for the procedure
        @qml.qnode(self.device)
        def unclamped_cost_function(p, logCF=False):
            self.unclamped_qaoa_circuit(p)
            if self.bitFlipNoise:
                [qml.BitFlip(0.05, wires=node) for node in range(self.num_total)]
            result = qml.expval(self.fullCostH)
            if logCF:
                self.CF.append(result.obs.data[0])
            return result

        # Use a classical optimizer to optimize the cost function circuit
        params = np.array(params, requires_grad=True)
        for i in range(self.optimizer_steps):
            if i == self.optimizer_steps - 1:
                params = self.optimizer.step(unclamped_cost_function, params, logCF=True)
            else:
                params = self.optimizer.step(unclamped_cost_function, params)

        # Return the now optimized params [gamma, nu]
        return params


    def sigmoid(self, x):
        """
        Sigmoid function, used in transforming data ----------------------------------------------
        :param x: The parameter to be plugged into the sigmoid function
        :return:
        """
        return 1.0 / (1.0 + np.exp(-x))


    def transform(self, data):
        """
        Transform the data to the machine -----------------------------------------------------
        :param data:                The data to be transformed
        :return:
        """
        # Special case if we only have one weight
        #if self.num_total == 2:
        #    w = self.WEIGHTS.item()
        #    result = self.sigmoid(np.dot(data, w))
        #else:
        #    result = self.sigmoid(np.dot(data, self.WEIGHTS))
        return self.sigmoid(np.dot(data, self.WEIGHTS))


    def train(self, DATA, unencodedData, learning_rate=0.1):
        """
        The training loop for the qRBM
        The unclamped sampling procedure is a subroutine of this larger loop
        :param DATA:
        :param unencodedData:
        :param learning_rate:
        :param num_epochs:
        :return:
        """
        # Measurement needed for weight update calculations
        @qml.qnode(self.device)
        def doublePZ(i, j):
            return qml.expval(qml.PauliZ(i) @ qml.PauliZ(j))

        # Timer for training process
        start = time.time()

        DATA = np.asarray(DATA)

        # ERRORS

        # Initial gamma and nu                                                                   UPDATE TO RANDOMIZE???
        gamma = np.random.rand()
        nu = np.random.rand()
        params = [gamma, nu]

        # Time limit == 5 hours
        timeout = time.time() + 60*60*5

        # Loop until distance is right
        hellinger_check = 1
        epoch_index = 0
        current_time = time.time()
        print("Start time: ", current_time)
        while (hellinger_check > self.stopping_criteria) and (current_time < timeout):
            # print("Beginning epoch ", epoch_index)
            new_weights = copy.deepcopy(self.WEIGHTS)
            neg_phase_quantum = np.zeros_like(self.WEIGHTS)

            # Run the unclamped sampling procedure to optimize gamma and nu
            params = self.unclamped_sampling(params)
            # Use the found params to calculate the negative phase
            if self.num_visible == 1 and self.num_hidden == 1:
                neg_phase_quantum = doublePZ(0, 0)
            else:
                for i in self.vis_indices:
                    for j in range(self.num_hidden):
                        neg_phase_quantum[i][j] = doublePZ(i, j)
            neg_phase_quantum *= (1. / float(len(DATA)))

            # Activation
            # special case nodes = 2
            #if self.num_total == 2:
            #    w = self.WEIGHTS.item()
            #    hidden_probs = self.sigmoid(np.dot(DATA, w))
            #    pos_hidden_states = hidden_probs > np.random.rand(len(DATA), self.num_hidden)
            #    neg_visible_activations = np.dot(pos_hidden_states, w)
            #    neg_visible_probs = self.sigmoid(neg_visible_activations)
            #    neg_hidden_activations = np.dot(neg_visible_probs, w)
            #else:
            hidden_probs = self.sigmoid(np.dot(DATA, self.WEIGHTS))
            pos_hidden_states = hidden_probs > np.random.rand(len(DATA), self.num_hidden)
            if self.num_visible == 1 and self.num_hidden == 1:
                neg_visible_activations = np.dot(pos_hidden_states, new_weights)
            else:
                neg_visible_activations = np.dot(pos_hidden_states, new_weights.T)
            neg_visible_probs = self.sigmoid(neg_visible_activations)
            neg_hidden_activations = np.dot(neg_visible_probs, new_weights)

            pos_phase = np.dot(DATA.T, hidden_probs) * (1. / float(len(DATA)))

            neg_hidden_probs = self.sigmoid(neg_hidden_activations)

            # Ommitted classical percentage for now --------------------------------------------------
            new_weights += learning_rate * (pos_phase - neg_phase_quantum)

            # Store the updated weights
            self.WEIGHTS = copy.deepcopy(new_weights)

            # Update the cost Hamiltonian
            self.updateHamiltonians()

            # Error tracking
            # Transform data to the hidden layer
            temp_transformed = self.transform(DATA)
            # Get error for each data point
            # for i in range(len(temp_transformed)):
            #    diff = abs(temp_transformed[i] - unencodedData[i])
            #    self.errors[epoch_index][i] = diff

            # Calculate Hellinger distance over all data points for this epoch
            hsum = 0
            for i in range(len(temp_transformed)):
                temp = (np.sqrt(unencodedData[i]) - np.sqrt(temp_transformed[i])) ** 2         # [i][0]
                hsum = hsum + temp
            hsum = np.sqrt(hsum)
            hsum *= (1 / np.sqrt(2))
            self.hellinger.append(hsum)
            hellinger_check = hsum

            epoch_index += 1
            current_time = time.time()

        # End of training loop
        end = time.time()

        # Export Hellinger data
        hdata = []
        # Extract the values from the objects in the hellinger list
        for value in self.hellinger:
            hdata.append(value.item())
        with open("hellinger_results.txt", 'w') as f:
            json.dump(hdata, f, indent=4)

        # Extract the values from the cost function list
        CFdata = []
        for value in self.CF:
            CFdata.append(value.item())
        with open("costfunction_results.txt", 'w') as f:
            json.dump(CFdata, f, indent=4)

        # Store the final values
        tdata = []
        for value in temp_transformed:
            tdata.append(value.item())
        with open("final_transformed.txt", 'w') as f:
            json.dump(tdata, f, indent=4)

        print("Training done in ", epoch_index, " epochs")
        print("Time taken for training: ", end - start, " seconds")
        #print(self.hellinger)




