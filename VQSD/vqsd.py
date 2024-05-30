# %%
import numpy as np
import pennylane as qml 
from pennylane import numpy as np
import matplotlib.pyplot as plt
import time
from qiskit_ibm_provider import IBMProvider
from qiskit.providers.fake_provider import *
import json
from tqdm import tqdm

# %%
def configured_backend():
    backend = FakeManila()
    return backend

# %%
class VQA: 

    def __init__(self, qubits, shots, trial):
        self.qubits = qubits
        self.shots = shots
        self.trial = trial

class VQSD(VQA):
    def __init__(self, qubits, shots, trial):
        super().__init__(qubits, shots, trial)
        #self.depth = depth

        self.dev = qml.device("qiskit.remote", wires=self.qubits*2, backend=configured_backend(), shots=shots) # device for real IBM devices 

    def run(self, param = None):
        # if param == None:
        #     param = []
        #     d = 1
        #     L = self.qubits + 2*(self.qubits - 1)*d
        #     for i in range(L):
        #         param.append(i)
        #print(param)
        
        def test_prep(param):
            for i in range(2*self.qubits):
                qml.Hadamard(i)
        
        def cost_cirq(param):
            for i in range(self.qubits):
                qml.CNOT(wires = [i,i+self.qubits])


        def rot(theta, i):
            qml.RZ(theta, wires = i)
            qml.RX(np.pi/2, wires = i)

        def initial_layer(param, q):
            for i in range(self.qubits):
                rot(param[i], i+q)

        def rot_ent(param, i, j):
            qml.CNOT(wires = [i, i+1])
            rot(param[j], i)
            rot(param[j+1], i+1)

        def rot_block(param, q):
            e = []
            o = []
            for i in range(self.qubits - 1):
                if i % 2 == 0:
                    e.append(i)
                    rot_ent(param, i + q, self.qubits + i)
                else:
                    o.append(i)
                
            for k in range(len(o)):
                rot_ent(param, o[k] + q, self.qubits + e[-1] + 2*(k+1))

        def Ansatz_block(param, q):
            initial_layer(param, q)
            rot_block(param, q)
            
        def Ansatz(param):
            Ansatz_block(param, 0)
            Ansatz_block(param, self.qubits)

        @qml.qnode(self.dev, interface="autograd")
        def vqsd(param):
            test_prep(param)
            Ansatz(param)
            cost_cirq(param)
            return qml.probs(wires = [i for i in range(self.qubits, 2*self.qubits)])

        drawer = qml.draw(vqsd)
        # print(drawer((param)))
        #print(drawer((param)), end='\r')

        def output(param):
            return 1-vqsd(param)[0] 

        return (output(param))
#
    def eval_readout(self, param = None):
        #if param == None:
        #    param = []
        #    d = 1
        #    L = self.qubits + 2*(self.qubits - 1)*d
        #    for i in range(L):
        #        param.append(1.57)

        def opt_test_prep(param):
            for i in range(self.qubits):
                qml.Hadamard(i)
                
        def rot(theta, i):
            qml.RZ(theta, wires = i)
            qml.RX(np.pi/2, wires = i)

        def initial_layer(param, q):
            for i in range(self.qubits):
                rot(param[i], i+q)

        def rot_ent(param, i, j):
            qml.CNOT(wires = [i, i+1])
            rot(param[j], i)
            rot(param[j+1], i+1)

        def rot_block(param, q):
            e = []
            o = []
            for i in range(self.qubits - 1):
                if i % 2 == 0:
                    e.append(i)
                    rot_ent(param, i + q, self.qubits + i)
                else:
                    o.append(i)
                
            for k in range(len(o)):
                rot_ent(param, o[k] + q, self.qubits + e[-1] + 2*(k+1))

        def Ansatz_block(param, q):
            initial_layer(param, q)
            rot_block(param, q)
            
        def Ansatz(param):
            Ansatz_block(param, 0)
            Ansatz_block(param, self.qubits)

        # dev2 = qml.device("default.mixed", wires = self.qubits, shots = self.shots)
        @qml.qnode(self.dev, interface = "autograd")
        def eval_read(param):
            opt_test_prep(param)
            Ansatz_block(param, 0)
            return qml.probs()

        return (eval_read(param))

    def opt(self, step, theta = None):
        if theta == None:
            theta = 0
            angle = []
            t = []
            d = 1
            L = self.qubits + 2*(self.qubits - 1)*d
            for i in range(L):
                t.append(2*(np.pi)*(np.random.uniform()))
            theta = np.array(t, requires_grad=True)
            angle = [theta]

        else:
            angle = theta
        
        error = []
        diag = []
        Ev = []
        system_size = 2*self.qubits

        diag = [self.run(theta)]
        Ev = [self.eval_readout(theta)]

        opt = qml.GradientDescentOptimizer(stepsize=step)
        #try ADAM
        max_iterations = 1000000000
        max_time = 18000
        conv_tol = 1.6e-03
        start_time = time.time()

        print("\n" f"initial value of the circuit parameter = {angle[0]}")

        for n in tqdm(range(max_iterations)):
            theta, prev_diag = opt.step_and_cost(self.run, theta)

            diag.append(self.run(theta))
            angle.append(theta)
            Ev = np.vstack((Ev, self.eval_readout(theta)))
            conv = 1 - max(Ev[-1,:])
            error.append(conv.numpy())
            Time = time.time() - start_time

            if n % 10 == 0:
                print(f"Step = {n},  Diagonality = {diag[-1]:.8f}, Eigen_Values = {Ev[-1]}")
            
            #if n == max_iterations:
            #    print("\n"f"Max iterations = {max_iterations}")
            #    break

            if Time >= max_time:
                print("\n"f"Max iterations = {n}")
                break
    
            if round(conv.item(), 3) <= conv_tol:
                max_iterations = n
                print("\n"f"Max iterations = {n}")
                print(f"The max eigenvalue is {max(Ev[-1,:])}")
                print(f"The convergence is {conv}")
                print(f"The eigenvalue error is {round(conv.item(), 3)}")
                break
        
        Time = time.time() - start_time

        print("\n" f"Optimization runtime = {Time}s")
        print("\n" f"Optimal value of the circuit parameter = {angle[-1]}")
        print("\n" f"Eigenvalues are: {Ev[-1,:]}")

        fig = plt.figure()
        fig.set_figheight(5)
        fig.set_figwidth(12)
        
        data = {
            "n_qubits": system_size, 
            "cost_history": diag, 
            "error_history": str(error),
            "ansatz": f"Single Layer Simplified 2-Design",
            "cost": f"Global",
            "optimizer": f"{opt}",
            "step size": step,
            "error_threshold": conv_tol,
            "noise_model": 'none',
            "Time to Solution": Time,
            "Iterations to Solution": max_iterations,
            "Final Error": str(conv),
            "Optimal_weights": str(angle[-1]),
            "Initial_weights": str(angle[0])
            }
        
        with open(f"VQSD/data/fakemanila/VQSD_{self.qubits}_FakeManila_{self.trial}.json", 'a') as outfile:
           json.dump(data, outfile)

        #print(data)
        #with open(f'data_toy_{self.qubits}.json', 'a') as fp:
        #    fp.write(",")
        #    json.dump(data, fp)

        #if save == True:
        #    filename='VQSD_'+'{self.qubits}'+'noise_model.json'
        #    script_path = os.path.abspath(__file__)
            #save_path = script_path.replace()
        #    completename = os.path,join(script_path, filename)

        #    with open(completename, 'wb') as file:
        #        pickle.dump(data, file) 

        plt.plot(range(len(diag)), diag, "k", ls = "solid")
        for i in range(len(Ev[n,:])):
            plt.plot(range(len(diag)), Ev[:,i], 'r', ls="dashed")
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.savefig(f"VQSD/diagrams/VQSD_{self.qubits}_FakeManila_{self.trial}.png")

        #return(data)

# %%

for i in range(2,5):
    z = VQSD(2, 100, i)
    z.opt(0.1)
