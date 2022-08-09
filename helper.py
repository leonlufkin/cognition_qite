import numpy as np

sigma_id = np.array([[1, 0],  [ 0, 1]], dtype=np.complex128) # define individual Pauli operators
sigma_sx = np.array([[0, 1],  [ 1, 0]], dtype=np.complex128)
sigma_sy = np.array([[0, -1j],[1j, 0]], dtype=np.complex128)
sigma_sz = np.array([[1, 0],  [0, -1]], dtype=np.complex128)

def HS(M1, M2):
   """Hilbert-Schmidt-Product of two matrices M1, M2"""
   return (np.dot(M1.conjugate().transpose(), M2)).trace()

def c2s(c):
   """Return a string representation of a complex number c"""
   if c == 0.0:
      return "0"
   if c.imag == 0:
      return "%g" % c.real
   elif c.real == 0:
      return "%gj" % c.imag
   else:
      return "%g+%gj" % (c.real, c.imag)

def state_to_str(state: np.ndarray):
   out = ''.join(state.astype(str))
   return out

def str_to_state(str_state, num_qubits):
   out = [int(str_state[i]) for i in range(len(str_state))]
   out += [0] * (num_qubits - len(str_state))
   return np.array(out)

def state_to_ind(state: np.ndarray):
   n = state.shape[0]
   bases = 4 ** np.arange(n)
   ind = int(np.sum(state * bases))
   return ind

def ind_to_state(ind: int, num_qubits):
   state = np.zeros(num_qubits, dtype=int)
   for i in range(num_qubits):
      r = ind % (4 ** (i+1))
      q = r // (4 ** i)
      state[i] = int(q)
   return state