import numpy as np
from qiskit import QuantumCircuit
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from itertools import product
from tqdm import tqdm

from args import parse_args
from helper import HS, c2s, state_to_str, str_to_state, state_to_ind, ind_to_state, sigma_id, sigma_sx, sigma_sy, sigma_sz
from index import idx, coeff

class QITE:
   def __init__(self, args, hamiltonian):
      # Quantum Circuit
      self.num_qubits = args.num_qubits
      assert hamiltonian.shape[0] == hamiltonian.shape[1] == 2 ** self.num_qubits == 2

      # Simulation
      self.num_steps = args.num_steps
      self.shots = args.shots
      self.db = args.db
      self.delta = args.delta
      self.simulator = QasmSimulator() # Use Aer's qasm_simulator

      # System
      self.hamiltonian = hamiltonian
      self.id = sigma_id
      self.sx = sigma_sx
      self.sy = sigma_sy
      self.sz = sigma_sz
      self.idx = idx
      self.coeff = coeff
      self.hamiltonian_decomp = self.decompose_hamiltonian()
      self.sigma_states = np.array([[0],[1],[2],[3]])

      # Measurement
      self.sigma_expectations = np.zeros((4,) * self.num_qubits, dtype=complex)
      self.alist = []

   def str_to_state(self, str_state):
      return str_to_state(str_state, self.num_qubits)

   def decompose_hamiltonian(self):
      """
      Decompose Hermitian matrix H into linear combination of Pauli matrices
      Works for both 4x4 (2 qubits) and 16x16 (4 qubits) cases
      """
      hamiltonian_decomp = {}    # to store results
      S = [self.id, self.sx, self.sy, self.sz]
      labels = ['0', '1', '2', '3']
      # print("Tensor product decomposition of Hamiltonian:")
      
      if self.hamiltonian.shape[0] == 2: # 2x2 matrix for 1 qubit
         entries = np.zeros(4,dtype=complex)
         entries[0] = (self.hamiltonian[0,0] + self.hamiltonian[1,1])/2
         entries[1] = (self.hamiltonian[0,1] + self.hamiltonian[1,0])/2
         entries[2] = (self.hamiltonian[0,1] - self.hamiltonian[1,0])*1j/2
         entries[3] = (self.hamiltonian[0,0] - self.hamiltonian[1,1])/2
         for i in range(4):
            if entries[i] >= 1E-10:
               hamiltonian_decomp[str(i)] = entries[i]

      elif self.hamiltonian.shape[0] == 4: # 4x4 matrix for 2 qubits
         for i in range(4):
            for j in range(4):
                  label = labels[i]+',' + labels[j]
                  a_ij = 0.25 * HS(np.kron(S[i], S[j]), self.hamiltonian)
                  if abs(a_ij) >= 1E-10:
                     # print("%s\t*\t( %s )" % (c2s(a_ij), label))   # save as string
                     hamiltonian_decomp[label] = float(a_ij)
                     
      elif self.hamiltonian.shape[0] == 16: # 16x16 matrix for 4 qubits
         for i in range(4):
            for j in range(4):
                  for k in range(4):
                     for l in range(4):
                        label = labels[i]+',' + labels[j]+','+ labels[k] +',' + labels[l] 
                        a_ij = 1/16 * HS(np.kron(np.kron(S[i], S[j]),np.kron(S[k], S[l])), self.hamiltonian)
                        if abs(a_ij) >= 1E-10:
                           # print("%s\t*\t( %s )" % (c2s(a_ij), label))
                           hamiltonian_decomp[label] = float(a_ij)
      # print("~~~ End ~~~")
      return hamiltonian_decomp

   def propagate(self):
      qc = QuantumCircuit(self.num_qubits, self.num_qubits)
      for t in range(len(self.alist)):
         for gate in range(1,4):
            angle = np.real(self.alist[t][gate])
            if gate == 1:
               qc.u(angle,-np.pi/2,np.pi/2, 0) # RX(angle,qbits[0])
            elif gate == 2:
               qc.u(angle,0,0, 0)# RY(angle,qbits[0])
            elif gate == 3:
               qc.p(angle,0)
      return qc

   def measure(self, qc, sigma_state: np.ndarray):
      """0=I, 1=X, 2=Y, 3=Z"""
      assert self.num_qubits == 1

      # Building quantum circuit
      # qc = QuantumCircuit(self.num_qubits, self.num_qubits)
      idx = sigma_state[0]
      # print(f"idx: {idx}")
      if idx == 0:
         return 1
      if idx == 1:
         qc.h(0)
         qc.measure([0],[0])
      elif idx == 2:
         qc.u(np.pi/2,-np.pi/2,np.pi/2, 0) # rotation about x-axis by pi/2
         qc.measure([0],[0])
      elif idx == 3:
         qc.measure([0],[0])

      # Simulating
      exe = transpile(qc, self.simulator)
      result = self.simulator.run(exe, shots=self.shots).result()
      counts = result.get_counts(exe)
      # print(counts)
      probs = {}
      for state in ['0', '1']:
         if state in counts.keys():
            probs[state] = counts[state]/self.shots
         else:
            probs[state] = 0
      Z_expectation = probs['0'] - probs['1']
      return Z_expectation

   def update_sigma_expectation(self):
      # qc = self.propagate()
      for sigma_state in self.sigma_states:
         # qc_copy = qc.copy()
         qc = self.propagate()
         self.sigma_expectations[sigma_state] = self.measure(qc, sigma_state)

   def measure_energy(self):
      energy = 0
      for sigma_state, coef in self.hamiltonian_decomp.items():
         qc = self.propagate()
         energy += coef * self.measure(qc, self.str_to_state(sigma_state))
      return energy

   def update_alist(self, ham):
      # Obtain A[m]
      # Step 1: Obtain S matrix
      S = np.zeros([4**self.num_qubits,4**self.num_qubits], dtype=complex)
      for row_state in self.sigma_states:
         for col_state in self.sigma_states:
            row_ind = state_to_ind(row_state)
            col_ind = state_to_ind(col_state)
            idx_ind = np.array([self.idx[i,j] for i,j in zip(row_state, col_state)])
            coeffs = np.array([self.coeff[i,j] for i,j in zip(row_state, col_state)])
            coef = np.prod(coeffs)
            S[row_ind,col_ind] = self.sigma_expectations[idx_ind] * coef

      # Step 2: Obtain b vector
      b = np.zeros([4 ** self.num_qubits],dtype=complex)
      c = 1
      # print(hm[0])
      for sigma_state, coef in ham.items():
         # print('state:')
         # print(self.str_to_state(sigma_state))
         c -= 2*self.db * coef * self.sigma_expectations[self.str_to_state(sigma_state)]
      # print('here')
      sqrt_c = np.sqrt(c)
      print(f"sqrt_c: {sqrt_c}")
      for sigma_state in self.sigma_states:
         ind = state_to_ind(sigma_state)
         b[ind] += self.sigma_expectations[sigma_state]*(1/sqrt_c-1)/(self.db)
         for ham_sigma_state, ham_coef in ham.items():
            ham_state = self.str_to_state(ham_sigma_state)
            print("sigma_state: ", end='')
            print(sigma_state)
            print("ham_state: ", end='')
            print(ham_state)
            idx_ind = np.array([self.idx[i,j] for i,j in zip(sigma_state, ham_state)])
            print(f"idx_ind: {idx_ind}")
            coeffs = np.array([self.coeff[i,j] for i,j in zip(sigma_state, ham_state)])
            coef = np.prod(coeffs)
            print(f"coef: {coef}")
            print(f"sigma_expectation: {self.sigma_expectations[idx_ind]}")
            b[ind] -= ham_coef * coef * self.sigma_expectations[idx_ind] / sqrt_c
         print(f"b[{ind}] = {b[ind]}")
         b[ind] = 1j*b[ind] - 1j*np.conj(b[ind])
         print(f"b[{ind}] = {b[ind]}")

      # Step 3: Add regularizer
      dalpha = np.eye(4) * self.delta

      # print(b)

      # Step 4: Solve for linear equation, the solution is multiplied by -2 because of the definition of unitary rotation gates is exp(-i theta/2)
      x = np.linalg.lstsq(S+np.transpose(S)+dalpha,-b,rcond=-1)[0]
      self.alist.append([])
      for i in range(len(x)):
         self.alist[-1].append(-x[i]*2*self.db)

   def qite_step(self):
      for sigma_state, coef in self.hamiltonian_decomp.items():
         self.update_sigma_expectation()
         ham = {sigma_state: coef}
         self.update_alist(ham)

   def run(self):
      E = np.zeros([self.num_steps+1],dtype=complex)
      E[0] = self.measure_energy()

      # Qite main loop
      for i in tqdm(range(1,self.num_steps+1)):
         self.qite_step()
         E[i] = self.measure_energy()
      return E, self.alist

if __name__ == '__main__':
   args = parse_args()

   # Build task
   if args.task == "demo" and args.num_qubits == 1:
      hamiltonian = 1 * sigma_sx + 1 * sigma_sz 

   qite = QITE(args, hamiltonian)
   print(qite.hamiltonian_decomp)
   ground_energy, ground_state = qite.run()
   print(qite.alist)
   print(ground_energy)
   # print(ground_state)
