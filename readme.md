# Welcome to the VQA Benchmarking Project! (beta)

### How close are Near-term Intermediate Scale Quantum (NISQ) applications to quantum advantage? 
### How can we develop clear and accessible benchmarks of Variational Quantum Algorithm (VQA) performance that can be compared across applications and that are operationally meaningful to the widest possible audience of end-users?

#### This is an open-source repository developed with funding from the Center for Quantum Networks to address these questions via our application-based, oprationally meaningful benchmarks that endeavor to assess VQA performance in a series of noisy environments. 
#### Through our partnership with Metriq, we hope that community members will add their own implementations of the many different VQAs to more fully benchmark the performance of NISQ algorithms.

##### For detailed information on our novel benchmarking approach, please refer to (https://docs.google.com/presentation/d/1DIuKx0fu_Yc2wbYRbZcZio9xHG0sk4zKP9gaDoP9BS8/edit?usp=sharing) - arXiv preprint coming soon!

#### Users can add to this growing project repo by running one of our initial implementations of the Variational Quantum Linear Solver (VQLS), Variational Quantum State Diagonalization (VQSD) Algorithm, Adiabatically Assisted Variational Quantum Eigensolver (AAVQE), or the Quantum Approximate Boltzman Machine (QABoM). In addition, users can add to this repo by implementing a new VQA and benchmarking it's performance in the following set of noisy environments:
##### 1. Noiseless simulator
##### 2. Bit flip noise model (bit flip probability p = 0.05)
##### 3. Device inspired noise model (e.g. IBM's FakeManilla noise simulator)
##### 4. Noisy quantum devices (IBM, Rigetti, IonQ, etc.)
    
## Please keep in mind that this repo is currently in beta! 
### We welcome any and all feedback as we prepare for the official launch in the coming weeks.