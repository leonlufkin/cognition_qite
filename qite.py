import numpy as np
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from itertools import product
from tqdm import tqdm
import matplotlib.pyplot as plt

from args import parse_args
from helper import HS, c2s, state_to_str, str_to_state, state_to_ind, ind_to_state, sigma_id, sigma_sx, sigma_sy, sigma_sz, quarterize
from index import idx, coeff

class QITE:
   def __init__(self, args, hamiltonian):
      # Quantum Circuit
      self.num_qubits = args.num_qubits
      assert hamiltonian.shape[0] == hamiltonian.shape[1] == 2 ** self.num_qubits # == 2

      # Simulation
      self.num_steps = args.num_steps
      self.shots = args.shots
      self.db = args.db
      self.delta = args.delta
      self.simulator = QasmSimulator() # Use Aer's qasm_simulator

      # System
      self.hamiltonian = hamiltonian
      self.id = sigma_id # Pauli basis
      self.sx = sigma_sx
      self.sy = sigma_sy
      self.sz = sigma_sz
      self.idx = idx # multiplication table for Pauli matrices
      self.coeff = coeff # coefficients in multiplication table for Pauli matrices
      pauli_states = [[0,1,2,3]]
      sigma_states = np.stack(np.meshgrid(*(self.num_qubits*pauli_states)), -1).reshape(-1, self.num_qubits) # list(product(*(pauli_states*self.num_qubits)))
      if self.num_qubits > 1:
         sigma_states[:, [0,1]] = sigma_states[:, [1,0]]
      self.sigma_states = sigma_states # basis for space of matrices of size 2^num_qubits x 2^num_qubits
      # self.num_sigma_states = int( (2 ** (self.num_qubits-1)) * (2 ** self.num_qubits - 1) ) # <- only used if ignoring states with odd # Y_hat 
      self.num_sigma_states = 4 ** self.num_qubits
      self.hamiltonian_decomp = {}

      # Measurement
      self.sigma_expectations = np.zeros((4,) * self.num_qubits, dtype=complex)
      self.alist = []
      self.qlist = [] # self.num_qubits * [[]]
      self.use_qlist = False

      # Debug - useful for analyzing internals of update_alist
      self.b = np.zeros([self.num_sigma_states],dtype=complex)
      self.S = np.zeros([self.num_sigma_states, self.num_sigma_states],dtype=complex)
      self.sqrt_c = 0

   def str_to_state(self, str_state):
      return str_to_state(str_state, self.num_qubits)

   def state_to_str(self, state):
      return state_to_str(state)

   def kron_list(self, kron_list):
      """computes Kronecker product of a list of matrices"""
      out = np.eye(1)
      for mat in kron_list:
         out = np.kron(out, mat)
      return out

   def decompose_hamiltonian(self):
      """
      Decompose Hermitian matrix H into linear combination of Pauli matrices
      Works for both 4x4 (2 qubits) and 16x16 (4 qubits) cases
      Based on Yu Jun Shen's code
      """
      hamiltonian_decomp = {}    # to store results
      S = [self.id, self.sx, self.sy, self.sz]
      norm_factor = 1/ (2 ** self.num_qubits)
      for state in self.sigma_states:
         label = self.state_to_str(state)
         a_ij = norm_factor * HS(self.kron_list([S[i] for i in state]), self.hamiltonian)
         if abs(a_ij) >= 1E-10:
            hamiltonian_decomp[label] = float(a_ij)
      return hamiltonian_decomp

   def propagate_alist(self):
      """Build circuit based on alist parameters"""
      qc = QuantumCircuit(self.num_qubits, self.num_qubits)
      for t in range(len(self.alist)):
         x = self.alist[t]
         for ind, sigma_state in enumerate(self.sigma_states):
            angle = x[ind]
            # if np.abs(angle) <= 1e-10:
            #    continue
            for qubit, gate in enumerate(sigma_state):
               if gate == 1:
                  qc.rx(angle, qubit) # RX(angle,qbits[0])
               elif gate == 2:
                  qc.ry(angle, qubit) # RY(angle,qbits[0])
               elif gate == 3:
                  qc.rz(angle, qubit) # RZ(angle,qbits[0])
      return qc

   def propagate_qlist(self):
      """Build circuit based on qlist parameters"""
      qc = QuantumCircuit(self.num_qubits, self.num_qubits)
      for entry in self.qlist:
         for qubit, angles in enumerate(entry):
            for i, angle in enumerate(angles):
               gate = i % 4
               if gate == 1:
                  qc.rx(angle, qubit) # RX(angle,qbits[0])
               elif gate == 2:
                  qc.ry(angle, qubit) # RY(angle,qbits[0])
               elif gate == 3:
                  qc.rz(angle, qubit) # RZ(angle,qbits[0])
      return qc

   def propagate(self):
      """Build circuit for current state"""
      if self.use_qlist:
         return self.propagate_qlist()
      else:
         return self.propagate_alist()

   def measure(self, qc, sigma_state: np.ndarray):
      """0=I, 1=X, 2=Y, 3=Z"""
      # If all identity gate applied to all qubits, then skip
      if sum(sigma_state) == 0:
         return 1

      # Building quantum circuit
      for qubit in range(self.num_qubits):
         idx = sigma_state[qubit]
         if idx == 1:
            qc.h(qubit)
         elif idx == 2:
            qc.u2(0, np.pi/2, qubit) # rotation about x-axis by pi/2 - WAS: qc.u(np.pi/2,-np.pi/2,np.pi/2, qubit)
      qubit_list = np.arange(self.num_qubits).tolist()
      qc.measure(qubit_list, qubit_list)

      # Simulating
      exe = transpile(qc, self.simulator)
      result = self.simulator.run(exe, shots=self.shots).result()
      counts = result.get_counts(exe)

      # Calculating probs
      probs = {}
      states = [np.binary_repr(i, width=self.num_qubits) for i in np.arange(2 ** self.num_qubits)]
      for state in states:
         if state in counts.keys():
            probs[state] = counts[state]/self.shots
         else:
            probs[state] = 0

      # Calculating expectation
      sign = {state: (-1) ** (state.count('1') % 2) for state in states}
      Z_expectation = 0
      for state in states:
         Z_expectation += sign[state] * probs[state]
      return Z_expectation

   def update_sigma_expectation(self):
      """Calculate expectation value for all basis matrices"""
      for sigma_state in self.sigma_states:
         qc = self.propagate()
         self.sigma_expectations[tuple(sigma_state)] = self.measure(qc, sigma_state)

   def measure_energy(self):
      """Measure energy of current state"""
      energy = 0
      for sigma_state, coef in self.hamiltonian_decomp.items():
         qc = self.propagate()
         energy += coef * self.measure(qc, self.str_to_state(sigma_state))
      return energy

   def update_alist(self, ham, verbose=False):
      """Update A[m] parameters to get next state"""
      # Obtain A[m]
      # Step 1: Obtain S matrix
      self.S = np.zeros([self.num_sigma_states, self.num_sigma_states], dtype=complex)
      for row_ind, row_state in enumerate(self.sigma_states):
         for col_ind, col_state in enumerate(self.sigma_states):
            idx_ind = tuple([self.idx[i,j] for i,j in zip(row_state, col_state)])
            coeffs = np.array([self.coeff[i,j] for i,j in zip(row_state, col_state)])
            coef = np.prod(coeffs)
            self.S[row_ind,col_ind] = self.sigma_expectations[idx_ind] * coef

      if verbose:
         print(self.S)

      # Step 2: Obtain b vector
      self.b = np.zeros([self.num_sigma_states],dtype=complex)
      c = 1
      for sigma_state, coef in ham.items():
         c -= 2*self.db * coef * self.sigma_expectations[self.str_to_state(sigma_state)]
      self.sqrt_c = sqrt_c = np.sqrt(c)

      if verbose:
         print(self.sqrt_c)

      # print(f"sqrt_c: {sqrt_c}")
      for ind, sigma_state in enumerate(self.sigma_states):
         if verbose:
            print(f"sigma_state: {tuple(sigma_state)}")
         # ind = state_to_ind(sigma_state)
         self.b[ind] += self.sigma_expectations[tuple(sigma_state)]*(1/sqrt_c-1)/(self.db)
         for ham_sigma_state, ham_coef in ham.items():
            ham_state = self.str_to_state(ham_sigma_state)
            idx_ind = tuple([self.idx[i,j] for i,j in zip(sigma_state, ham_state)])
            
            if verbose:
               print(f"idx_ind: {idx_ind}")

            coefs = np.array([self.coeff[i,j] for i,j in zip(sigma_state, ham_state)])
            if verbose:
               print(coefs)
            coef = np.prod(coefs)

            if verbose:
               print(f"ham_coef: {ham_coef}")
               print(f"coef: {coef}")
               print(f"self.sigma_expectations[idx_ind] / sqrt_c: {self.sigma_expectations[idx_ind] / sqrt_c}")
               print(f"-=: {ham_coef * coef * self.sigma_expectations[idx_ind] / sqrt_c}")
            self.b[ind] -= ham_coef * coef * self.sigma_expectations[idx_ind] / sqrt_c

      if verbose:
         print(self.b)

      self.b = -2 * self.b.imag

      if verbose:
         print(self.b)

      # Step 3: Add regularizer
      dalpha = np.eye(self.num_sigma_states) * self.delta

      # Step 4: Solve for linear equation, the solution is multiplied by -2 because of the definition of unitary rotation gates is exp(-i theta/2)
      x = np.linalg.lstsq(self.S+np.transpose(self.S)+dalpha,-self.b,rcond=-1)[0]
      self.alist.append((-(x.real)*2*self.db).tolist())

      if verbose:
         print(x)

   def update_qlist(self):
      """Condense A[m] parameters; provides a speed-up for num_qubits > 1 by reducing circuit size"""
      # x = np.expand_dims(self.alist[-1], 1)
      x = self.alist[-1]
      self.qlist.append([])
      for _ in range(self.num_qubits):
         x = quarterize(x)
         comp = x.sum(-1).flatten()
         self.qlist[-1].append(comp)

   def qite_step(self):
      """One step in QITE"""
      for sigma_state, coef in self.hamiltonian_decomp.items():
         self.update_sigma_expectation()
         ham = {sigma_state: coef}
         self.update_alist(ham)
         if self.use_qlist:
            self.update_qlist()

   def run(self, use_qlist = False):
      """QITE algorithm"""
      self.use_qlist = use_qlist
      self.hamiltonian_decomp = self.decompose_hamiltonian()
      E = np.zeros([self.num_steps+1],dtype=complex)
      E[0] = self.measure_energy()

      # Qite main loop
      for i in tqdm(range(1,self.num_steps+1)):
         self.qite_step()
         E[i] = self.measure_energy()
      return E, self.alist

   def run_from_alist(self, alist):
      """Calculate energy along sequences of states given by alist"""
      my_alist = self.alist
      n_steps = len(alist)
      E = np.zeros(n_steps+1)
      for step in range(n_steps):
         self.alist = alist[:step]
         E[step+1] = self.measure_energy()
      self.alist = my_alist
      return E


if __name__ == '__main__':
   args = parse_args()

   # Build task
   if args.task == "demo":
      if args.num_qubits == 1:
         hamiltonian = 1/np.sqrt(2) * sigma_sx + 1/np.sqrt(2) * sigma_sz
      elif args.num_qubits == 2:
         hamiltonian = np.kron(sigma_id, sigma_sz) # np.kron(sigma_sx, sigma_sz) + np.kron(sigma_sy, sigma_sz)
      elif args.num_qubits == 3:
         hamiltonian = np.kron(np.kron(sigma_id, sigma_id), sigma_sx)

   qite = QITE(args, hamiltonian)
   print(qite.hamiltonian_decomp)
   print(np.linalg.eig(hamiltonian)[0].min())
   qc = qite.propagate()
   qc.draw(output='mpl')
   plt.savefig('qite_circuit0.png')
   plt.close()
   ground_energy, ground_state = qite.run()
   qc = qite.propagate()
   qc.draw(output='mpl')
   plt.savefig('qite_circuit.png')
   plt.close()
   print(np.array(qite.alist).shape)
   print(ground_energy)
   plt.plot(np.arange(args.num_steps+1) * args.db, ground_energy)
   plt.savefig('qite_evolution.png')
   plt.close()
