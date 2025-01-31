import mitiq
import cirq
import numpy as np
from mitiq import QPROGRAM, Executor, Observable, QuantumResult, MeasurementResult
from mitiq.executor.executor import DensityMatrixLike, MeasurementResultLike
from typing import Callable, Optional, Union, Sequence, List
import time



# This virtual distillation works only for M = 2 copies of the state rho
M = 2

Z = np.array([[1, 0], [0, -1]])
X = np.array([[0, 1], [1, 0]])

def M_copies_of_rho(rho: cirq.Circuit, M: int=2) -> cirq.Circuit:
    '''
    Given a circuit rho that acts on N qubits, this function returns a circuit that copies rho M times in parallel.
    This means the resulting circuit has N * M qubits.

    Args:
        rho: The input circuit rho acting on N qubits
        M: The number of copies of rho

    Returns:
        A circuit that copies rho M times in parallel.
    '''

    N = len(rho.all_qubits())

    circuit = cirq.Circuit()
    qubits = cirq.LineQubit.range(N*M)

    for i in range(M):
        circuit += rho.transform_qubits(lambda q: qubits[q.x + N*i])

    return circuit

def diagonalize(U: np.ndarray) -> np.ndarray:
    """
    Diagonalize a density matrix rho and return the basis change unitary V†.

    Args:
        U: The density matrix to be diagonalized.
    
    Returns:
        V†: The basis change unitary.
    """
    
    eigenvalues, eigenvectors = np.linalg.eigh(U)
    
    # Sort eigenvalues and eigenvectors by ascending phase
    phases = np.angle(eigenvalues)
    sorted_indices = np.argsort(phases)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]
    
    # Normalize and enforce sign convention (optional)
    for i in range(sorted_eigenvectors.shape[1]):
        # Force the first nonzero element of each eigenvector to be positive
        if np.sign(sorted_eigenvectors[:, i][0]) < 0:
            sorted_eigenvectors[:, i] *= -1
    
    # Compute V† (conjugate transpose of V)
    V_dagger = np.conjugate(sorted_eigenvectors.T)
    
    # check
    if not np.allclose(U, np.dot(sorted_eigenvectors, np.dot(np.diag(sorted_eigenvalues), V_dagger))):
        raise ValueError("Diagonalization failed.")

    return V_dagger, sorted_eigenvalues

def failed_attempt_to_optimise_execute_with_vd(input_rho: cirq.Circuit, M: int=2, K: int=100, observable=Z):
    '''
    Given a circuit rho that acts on N qubits, this function returns the expectation values of a given observable for each qubit i. 
    The expectation values are corrected using the virtual distillation algorithm. 
    '''

    # input rho is an N qubit circuit
    N = len(input_rho.all_qubits())
    rho = M_copies_of_rho(input_rho, M)

    # Coupling unitary corresponding to the diagonalization of the SWAP (as seen in the paper) for M = 2:
    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])

    Ei = [0 for _ in range(N)]
    D = 0
    
    # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K -= 1

    # Helper function to map the results to the eigenvalues of the pauli Z observable
    def map_to_eigenvalues(measurement):
        if measurement == 0:
            return 1
        else:
            return -1
        

    
    # prepare gates
    B_gate = cirq.MatrixGate(Bi_gate)

    need_basis_change = not np.allclose(observable, Z)
    if need_basis_change:
        basis_change_unitary = diagonalize(observable)[0]
        gate = cirq.MatrixGate(basis_change_unitary)
    else: 
        gate = None
    
    for _ in range(K):
        
        circuit = rho.copy()

        for i in range(N):
            # 1) apply basis change unitary to all M * N qubits
            if need_basis_change:
                for m in range(M):
                    circuit.append(gate(cirq.LineQubit(i + m*N)))

            # 2) apply the diagonalization gate B
            # [implementation works only for the M=2 case]
            circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+N)))

        
            # 3) apply measurements
            # the measurement keys are applied in accordance with the SWAPS that are applied in the pseudo code in the paper.
            # The SWAP operations are omitted here since they are hardware specific.
            # [once again this specific code is for M = 2]
            circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{2*i}"))
            circuit.append(cirq.measure(cirq.LineQubit(i+N), key=f"{2*i+1}"))
        
        # run the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
                
        # post processing measurements
        z1 = []
        z2 = []
        
        for i in range(2*N):
            if i % 2 == 0:
                z1.append(np.squeeze(result.records[str(i)]))
            else:
                z2.append(np.squeeze(result.records[str(i)]))
    
        z1 = [map_to_eigenvalues(i) for i in z1]
        z2 = [map_to_eigenvalues(i) for i in z2]

        # Part obtained from the pseudocode of the paper
        productD = 1
        for i in range(N):
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )
            Ei[i] += 1/2**N * (z1[i] + z2[i]) * productE

            productD *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )
        
        D += 1/2**N * productD 
    
    # K must be odd so that D cannot accidentally be 0 and give an error.
    # [Forcing odd K is a workaround, we should look into this]
    Z_i_corrected = [Ei[i] / D for i in range(N)]

    return Z_i_corrected

# NO LONGER NEEDED
def create_S2_N_matrix(N_qubits):
    """
    Function that creates a matrix corresponding to the cyclic swap operation of two N qubit circuits
    Input: N_qubits: number of qubits in one circuit
    output: np.array of the unitary quantum gate on 2*N qubits that swaps the circuits.    
    """
    M = 2 # change this to an argument once support for other values is added


    SWAP_matrix = np.array([
        [1,0,0,0],
        [0,0,1,0],
        [0,1,0,0],
        [0,0,0,1]
    ])

    swap_step_matrices = [0 for _ in range(N_qubits)]
    swap_step_matrices[0] = SWAP_matrix
    swap_step_matrices[-1] = SWAP_matrix


    # create the SWAP^tensor(i) matrix
    for i in range(1, N_qubits):
        swap_step_matrices[i] = np.kron(swap_step_matrices[i-1], SWAP_matrix)



    def add_swap_step(i, matrix):
               
        # Create the I^tensor(N-i)SWAP^tensor(i)I^tensor(N-i) matrix
        step_matrix = np.kron( np.eye(2**(N_qubits - i)), np.kron(swap_step_matrices[i-1], np.eye(2**(N_qubits - i)) ) )
        
        # Apply swapping step
        return step_matrix @ matrix


    S_matrix = np.eye(2 ** (M*N_qubits))
    
    # let i go up from 1 to N
    for i in range(1, N_qubits+1):
        S_matrix = add_swap_step(i, S_matrix)
    
    # let i go back down from N-1 to 1
    for i in range(N_qubits-1, 0,-1):
        S_matrix = add_swap_step(i, S_matrix)
    
    return S_matrix

def apply_cyclic_system_permutation(matrix, N_qubits, M=2):
    """
        Function that shifts the rows of a matrix or vector in such a way, 
        that each of the M registers of N_qubit qubits are shifted cyclically.
        The implementation is identical to left multiplication with repeated swap gates,
        however this optimisation in considerably faster. 
    """
    matrix = np.array(matrix)
    
    # determine the row permutation for the cyclic shift operation
    permutation = [j+i for j in range(2**N_qubits) for i in range(0, 2**(M*N_qubits), 2**N_qubits)]
    
    # Some fancy index magic to permute the rows in O(n) time and space (n=2**(M*N_qubits))
    idx = np.empty_like(permutation)
    idx[permutation] = np.arange(len(permutation))

    # allow to work with lists or arrays of np.ndarrays
    if matrix.ndim == 2:
        matrix = matrix[idx] 
    elif matrix.ndim == 3:
        matrix[:] = matrix[:, idx]
    else:
        raise TypeError("matrix must be a 2 dimensional array or a listor array of 2 dimensional arrays") 
    return matrix

def apply_symmetric_observable(matrix, N_qubits, observable = None, M=2):
    if observable == None or np.allclose(observable, np.array([[1., 0.], [0.,-1.]])):
        # use the default Z observable
        sym_observable_diagonals = []
        for i in range(N_qubits):
            observable_i_diagonal = np.array([ j for k in range(2**(i)) for j in [1., -1.] for l in range(2**(N_qubits-i-1))])
            
            # turn [a, b, c] into [a,a,a,b,b,b,c,c,c]. This is the same as tensoring the N_qubit identity on the right
            observable_i_diagonal_system1 = np.array([observable_i_diagonal for _ in range(2**N_qubits) ]).flatten('F')
            # turn [a,b,c] into [a,b,c,a,b,c,a,b,c]. This is the same as tensoring the N_qubit identity on the left
            observable_i_diagonal_system2 = np.array([observable_i_diagonal for _ in range(2**N_qubits) ]).flatten('C')
            # add the symmetric observable
            sym_observable_diagonals.append(0.5 * (observable_i_diagonal_system1 + observable_i_diagonal_system2))
        
        if matrix.ndim == 2:
            return np.array([sod * matrix for sod in sym_observable_diagonals])
        elif matrix.ndim == 3:
            return np.array([sod * mat for sod in sym_observable_diagonals for mat in matrix])
        return 

    else:
        sym_observable_matrices = []
        for i in range(N_qubits):
            observable_i_matrix = np.kron(np.kron(np.eye(2**i), observable), np.eye(2**(N_qubits-i-1)))
            sym_observable_matrix = 0.5 * (np.kron(observable_i_matrix, np.eye(2**N_qubits)) + np.kron(np.eye(2**N_qubits), observable_i_matrix))
            sym_observable_matrices.append(sym_observable_matrix)

    return np.array(sym_observable_matrices) @ matrix

class style():
    black = '\033[30m'
    red = '\033[31m'
    green = '\033[32m'
    yellow = '\033[33m'
    blue = '\033[34m'
    magenta = '\033[35m'
    cyan = '\033[36m'
    white = '\033[37m'
    BLACK = '\033[90m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    red_BG = '\033[41m'
    white_BG = '\033[47m'
    UNDERLINE = '\033[4m'
    RESET = '\033[0m'

def executor_execute_with_vd(
        circuit: QPROGRAM, 
        executor: Union[Executor, Callable[[QPROGRAM], QuantumResult]], 
        M: int=2, 
        K: int=100, 
        observable: Optional[Observable] = None,
        display_performance: bool = False
    ) -> list[float]:
    '''
    Given a circuit rho that acts on N qubits, this function returns the expectation values of a given observable for each qubit i. 
    The expectation values are corrected using the virtual distillation algorithm.

    Args:
        circuit: The input circuit of N qubits to execute with VD.
        executor: A Mitiq executor that executes a circuit and returns the
                    unmitigated ``QuantumResult`` (e.g. an expectation value).
                    The executor must either return a single measurement (bitstring or list),
                    a list of measurements
        M: The number of copies of rho. Only M=2 is implemented at this moment.
        K: The number of iterations of the algorithm. Only used if the executor returns a single measurement.
        observable: The one qubit observable for which the expectation values are computed. 
                    The default observable is the Pauli Z matrix.
                    At the moment using different observables is not supported.

    Returns:
        A list of expectation values for each qubit i in the circuit. Estimated with VD.
    '''
    T0 =time.time()
    timing_statements = []

    # input rho is an N qubit circuit
    N = len(circuit.all_qubits())
    rho = M_copies_of_rho(circuit, M)

    # Coupling unitary corresponding to the diagonalization of the SWAP (as seen in the paper) for M = 2:
    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])

    Ei = np.array(list(0 for _ in range(N)))
    D = 0
    
    # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K -= 1

    # Changing basis to accomodate different observables 
    # Out of scope for the mitiq-UvA project
            
        # # 1) apply basis change unitary
        # # for example observable Z -> apply I
        # # for example observable X -> apply H
        # basis_change_unitary = diagonalize(observable)[0]
        
        # # apply to every single qubit
        # if not np.allclose(basis_change_unitary, np.eye(2)):
        #     gate = cirq.MatrixGate(basis_change_unitary)
        #     for i in range(M*N):
        #         rho_copy.append(gate(cirq.LineQubit(i)))

    
    # 2) apply the diagonalization gate B
    # once again this specific code is for M = 2
    B_gate = cirq.MatrixGate(Bi_gate)
    for i in range(N):
        rho.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+N)))
    
    if not isinstance(executor, Executor):
        executor = Executor(executor)
    T1 =time.time()
    timing_statements.append(f"{style.white}{'Preparation step':30s} ~ {T1-T0:8.3f} s{style.RESET}")
    
    if executor._executor_return_type in DensityMatrixLike:
        # do density matrix treatment

        rho_tensorM = executor.run(rho)
        T2 =time.time()
        timing_statements.append(f"{style.red}{'Simulation step':30s} ~ {T2-T1:8.3f} s{style.RESET}")
        # print(f"rho^tensor M:\n{rho_tensorM}")
       
        # two_system_swap = create_S2_N_matrix(N)
        rho_tensorM_swapped = apply_cyclic_system_permutation(rho_tensorM, N)
        T3 =time.time()
        timing_statements.append(f"{style.yellow}{'Apply cyclic swap':30s} ~ {T3-T2:8.3f} s{style.RESET}")
        # print(f"S2_N:\n{two_system_swap}")

        rho_tensorM_swapped_observabled = apply_symmetric_observable(rho_tensorM_swapped, N, observable)
        T4 =time.time()
        timing_statements.append(f"{style.cyan}{'Observable creation':30s} ~ {T4-T3:8.3f} s{style.RESET}")
        # print(f"O_i sym:\n{symmetric_Obs_i}")

        Z_i_corrected = np.trace(rho_tensorM_swapped_observabled, axis1=1, axis2=2) / np.trace(rho_tensorM_swapped, axis1=1, axis2=2)
        
        T5 =time.time()
        timing_statements.append(f"{style.green}{'Computation step':30s} ~ {T5-T4:8.3f} s{style.RESET}")

        timing_statements.append(f"Total time: {T5-T0:8.3f} sec. |{style.white}{int(100*(T1-T0)/(T5-T0))*'#'}{style.red}{int(100*(T2-T1)/(T5-T0))*'#'}{style.cyan}{int(100*(T4-T3)/(T5-T0))*'#'}{style.yellow}{int(100*(T3-T2)/(T5-T0))*'#'}{style.green}{int(100*(T5-T4)/(T5-T0))*'#'}{style.RESET}| {style.white}{(T1-T0)/(T5-T0):6.3%} {style.red}{(T2-T1)/(T5-T0):6.3%} {style.cyan}{(T4-T3)/(T5-T0):6.3%} {style.yellow}{(T3-T2)/(T5-T0):6.3%} {style.green}{(T5-T4)/(T5-T0):6.3%}{style.RESET}")
    

    elif executor._executor_return_type in MeasurementResultLike:

        #  3) apply measurements
        # The measurements are only added when the executor returns measurement values
        # the measurement keys are applied in accordance with the SWAPS that are applied in the pseudo code in the paper.
        # The SWAP operations are omitted here since they are hardware specific.
        for i in range(M*N):
            rho.append(cirq.measure(cirq.LineQubit(i), key=f"{i}"))

        T2 =time.time()
        timing_statements.append(f"{style.cyan}{'added measurements':30s} ~ {T2-T1:8.3f} s{style.RESET}")

        # if executor._executor_return_type ==  MeasurementResult: # !!!!!!!!!!!!! aaaaaaaaaasagagahgahghaaagaagahaahahgahggggggghghgghghggh
        
        res = executor.run(rho, force_run_all=True, reps=K) # TODO make this reps a **kwargs to allow any executor
        # print(f"{res=}")

        self_packed = True
        if isinstance(res, str):
            # print("was string")
            res = [res]
            self_packed = False
        elif isinstance(res[0], int):
            # print("was int list")
            res = [res]
            self_packed = False

        if len(res) == 1: # if the executor only returns a single measurement
            for _ in range(K-1): # then we measure K times in total
                if not self_packed:                    
                    res.append( executor.run(rho, force_run_all=True))
                else:
                    res.append( executor.run(rho, force_run_all=True)[0] )

        T3 =time.time()
        timing_statements.append(f"{style.red}{'Running executor':30s} ~ {T3-T2:8.3f} s{style.RESET}")

        # post processing measurements
        for bitStr in res:
            
            # This maps 0/1 measurements to 1/-1 measurements, the eigenvalues of the Z observable
            Z_base_mesurement = 1 - 2* np.array(list(bitStr))

            # Separate the two systems
            z1 = Z_base_mesurement[:N]
            z2 = Z_base_mesurement[N:]

            # Implementing the sum and product from the paper
            # Note that integer division prevents floating point errors here, 
            # since each factor in the product or the Ei sum will be either +1 or -1.
            product_term = 1
            for j in range(N):
                product_term *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )//2

            D += product_term 

            for i in range(N):
                Ei[i] += (z1[i] + z2[i])//2 * product_term // (( 1 + z1[i] - z2[i] + z1[i]*z2[i] )//2) # undo the j=i term in the product

        T4 =time.time()
        timing_statements.append(f"{style.yellow}{'Postprocessing step':30s} ~ {T4-T3:8.3f} s{style.RESET}")

        # else: # the K shots are already performed by the executor 
        #     raise ValueError("We do not have support for running multiple circuits at once, at this moment.")

        # Elementwise division by D, since we are working with numpy arrays
        Z_i_corrected = Ei / D
        T5 =time.time()
        timing_statements.append(f"{style.green}{'Computation step':30s} ~ {T5-T4:8.3f} s{style.RESET}")

        timing_statements.append(f"Total time: {T5-T0:8.3f} sec. |{style.white}{int(100*(T1-T0)/(T5-T0))*'#'}{style.cyan}{int(100*(T2-T1)/(T5-T0))*'#'}{style.red}{int(100*(T3-T2)/(T5-T0))*'#'}{style.yellow}{int(100*(T4-T3)/(T5-T0))*'#'}{style.green}{int(100*(T5-T4)/(T5-T0))*'#'}{style.RESET}| {style.white}{(T1-T0)/(T5-T0):6.3%} {style.cyan}{(T2-T1)/(T5-T0):6.3%} {style.red}{(T3-T2)/(T5-T0):6.3%} {style.yellow}{(T4-T3)/(T5-T0):6.3%} {style.green}{(T5-T4)/(T5-T0):6.3%}{style.RESET}")
    

    else:
        raise ValueError("Executor must have a return type of DensityMatrixLike or MeasurementResultLike")


    if display_performance:
        for line in timing_statements:
            print(line)

    if not np.allclose(Z_i_corrected.real, Z_i_corrected, atol=1e-6):
        print("Warning: The expectation value contains a significant imaginary part. This should never happen.")
        return Z_i_corrected
    else:
        return Z_i_corrected.real

def execute_with_vd(input_rho: cirq.Circuit, M: int=2, K: int=100, observable=Z) -> list[float]:
    '''
    Given a circuit rho that acts on N qubits, this function returns the expectation values of a given observable for each qubit i. 
    The expectation values are corrected using the virtual distillation algorithm.

    Args:
        input_rho: The input circuit rho acting on N qubits
        M: The number of copies of rho
        K: The number of iterations of the algorithm
        observable: The observable for which the expectation values are computed. 
                    The default observable is the Pauli Z matrix.

    Returns:
        A list of expectation values for each qubit i in the circuit.
    '''

    # input rho is an N qubit circuit
    N = len(input_rho.all_qubits())
    rho = M_copies_of_rho(input_rho, M)

    # Coupling unitary corresponding to the diagonalization of the SWAP (as seen in the paper) for M = 2:
    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])

    Ei = [0 for _ in range(N)]
    D = 0
    
    # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K -= 1

    for _ in range(K):
        
        circuit = rho.copy()

        # 1) apply basis change unitary
        # for example observable Z -> apply I
        # for example observable X -> apply H
        basis_change_unitary = diagonalize(observable)[0]
        
        # apply to every single qubit
        if not np.allclose(basis_change_unitary, np.eye(2)):
            gate = cirq.MatrixGate(basis_change_unitary)
            for i in range(M*N):
                circuit.append(gate(cirq.LineQubit(i)))

        # 2) apply the diagonalization gate B
        B_gate = cirq.MatrixGate(Bi_gate)
        for i in range(N):
            circuit.append(B_gate(cirq.LineQubit(i), cirq.LineQubit(i+N)))

        
        # 3) apply measurements
        # the measurement keys are applied in accordance with the SWAPS that are applied in the pseudo code in the paper.
        # The SWAP operations are omitted here since they are hardware specific.
        # once again this specific code is for M = 2
        for i in range(N):
            circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{2*i}"))
        for i in range(N):
            circuit.append(cirq.measure(cirq.LineQubit(i+N), key=f"{2*i+1}"))
        
        # run the circuit
        simulator = cirq.Simulator()
        result = simulator.run(circuit, repetitions=1)
                
        # post processing measurements
        z1 = []
        z2 = []
        
        for i in range(2*N):
            if i % 2 == 0:
                z1.append(np.squeeze(result.records[str(i)]))
            else:
                z2.append(np.squeeze(result.records[str(i)]))

        # this one is for the pauli Z obvservable
        def map_to_eigenvalues(measurement):
            if measurement == 0:
                return 1
            else:
                return -1
            
        z1 = [map_to_eigenvalues(i) for i in z1]
        z2 = [map_to_eigenvalues(i) for i in z2]

        for i in range(N):
            
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )

            Ei[i] += 1/2**N * (z1[i] + z2[i]) * productE

        productD = 1
        for j in range(N):
            productD *= ( 1 + z1[j] - z2[j] + z1[j]*z2[j] )

        D += 1/2**N * productD 
        
    Z_i_corrected = [Ei[i] / D for i in range(N)]

    return Z_i_corrected

def optimized_execute_with_vd(input_circuit: cirq.Circuit, 
                              num_copies: int=2, 
                              K: int=1000, 
                              observable=None) -> list[float]:
    
    """ optimized """

    N = len(input_circuit.all_qubits())
    circuit_copies = M_copies_of_rho(input_circuit, num_copies)

    Bi_gate = np.array([
            [1, 0, 0, 0],
            [0, np.sqrt(2)/2, np.sqrt(2)/2, 0],
            [0, np.sqrt(2)/2, -np.sqrt(2)/2, 0],
            [0, 0, 0, 1]
        ])
    BGATE = cirq.MatrixGate(Bi_gate)

    Ei = [0 for _ in range(N)]
    D = 0

    if observable is not None:
        # TODO
        pass

    def map_to_eigenvalues(measurement):
        if measurement == 0:
            return 1
        else:
            return -1
        
    vd_circuit = circuit_copies.copy()

    # 1) apply the diagonalization gate B
    for i in range(N):
        vd_circuit.append(BGATE(cirq.LineQubit(i), cirq.LineQubit(i+N)))

    # 2) apply measurements
    for i in range(N):
        vd_circuit.append(cirq.measure(cirq.LineQubit(i), key=f"{2*i}"))
    for i in range(N):
        vd_circuit.append(cirq.measure(cirq.LineQubit(i+N), key=f"{2*i+1}"))

    simulator = cirq.Simulator()
    result = simulator.run(vd_circuit, repetitions=K)

     # Forcing odd K, this is a workaround so that D (see end of the function) cannot be 0 accidentally
    if K%2 == 0:
        K -= 1

    for k in range(K):
        z1 = []
        z2 = []
        
        # Extract measurements for repetition k
        for i in range(2 * N):
            measurement = np.squeeze(result.records[str(i)][k])  # Get the k-th repetition
            if i % 2 == 0:
                z1.append(measurement)
            else:
                z2.append(measurement)
        
        z1 = [map_to_eigenvalues(m) for m in z1]
        z2 = [map_to_eigenvalues(m) for m in z2]

        # Update Ei and D
        for i in range(N):
            productE = 1
            for j in range(N):
                if i != j:
                    productE *= (1 + z1[j] - z2[j] + z1[j] * z2[j])
            Ei[i] += 1 / 2**N * (z1[i] + z2[i]) * productE

        productD = 1
        for j in range(N):
            productD *= (1 + z1[j] - z2[j] + z1[j] * z2[j])
        D += 1 / 2**N * productD

    # Calculate the corrected Z_i
    Z_i_corrected = [Ei[i] / D for i in range(N)]

    return Z_i_corrected