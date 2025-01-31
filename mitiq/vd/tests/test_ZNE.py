# %%
import numpy as np
import cirq
import mitiq
from mitiq.zne.scaling import fold_gates_at_random
from mitiq.zne.inference import RichardsonFactory
import tqdm
import pickle
from benchmarking_funcs import *

# %%
def execute_with_zne(circuit, executor, scale_factors=[1, 2, 3]):
    """Executes a circuit with Zero Noise Extrapolation (ZNE)."""
    factory = RichardsonFactory(scale_factors=scale_factors)  # ✅ Pass scale_factors here
    mitigated_expval = mitiq.zne.execute_with_zne(
        circuit,
        executor=executor,
        factory=factory
    )
    return mitigated_expval

def noisy_executor(circuit, noise_level):
    """Executes the circuit with depolarizing noise."""
    noisy_circuit = circuit.with_noise(cirq.depolarize(p=noise_level))
    return np.mean(expectation_Zi(noisy_circuit))

def run_benchmarking_with_zne(run_name, vd_iterations, scale_factors, N_datapoints=10, N_qubits=6, N_layers=20, entangled=True):
    """Runs a benchmark for both Virtual Distillation and Zero Noise Extrapolation."""

    rho, gate_count, _ = create_randomised_benchmarking_circuit(N_qubits, N_layers, entanglement=entangled)
    
    # Ensure the circuit is meaningful
    true_Zi = expectation_Zi(rho)
    while np.all(true_Zi == 0. + 0.j):
        rho, gate_count, _ = create_randomised_benchmarking_circuit(N_qubits, N_layers, entanglement=entangled)
        true_Zi = expectation_Zi(rho)

    BMrun = {
        "benchmark_name": run_name,
        "observable": "Z",
        "N_qubits": N_qubits,
        "N_layers": N_layers,
        "rho": rho,
        "gate_count": gate_count
    }

    print(f"Running benchmark: {run_name}")

    # Setup progress bar
    total_steps = 1 + 1 + N_datapoints * (1 + sum(vd_iterations) + len(scale_factors))
    pbar = tqdm.tqdm(total=total_steps)

    datapoints = []
    datapoint_labels = ["Noisy"]

    for N_exp_Err in np.logspace(-2, 0, base=10, num=N_datapoints):
        noise_level = N_exp_Err / gate_count
        noisy_Zi = expectation_Zi(rho.with_noise(cirq.depolarize(p=noise_level)))
        dist_noisy_Zi = vector_norm_distance(true_Zi, noisy_Zi)
        pbar.update(1)

        measurement_list = [N_exp_Err, dist_noisy_Zi]

        # VD Execution
        for K in vd_iterations:
            vd_Zi = execute_with_vd(rho.with_noise(cirq.depolarize(p=noise_level)), 2, K)
            dist_vd_Zi = vector_norm_distance(true_Zi, vd_Zi)
            measurement_list.append(dist_vd_Zi)
            datapoint_labels.append(f"vd: {K=}")
            pbar.update(K)

        # ZNE Execution
        for scale in scale_factors:
            mitigated_Zi = execute_with_zne(rho, lambda c: noisy_executor(c, noise_level), scale_factors)
            dist_zne_Zi = vector_norm_distance(true_Zi, mitigated_Zi)
            measurement_list.append(dist_zne_Zi)
            datapoint_labels.append(f"zne: {scale=}")
            pbar.update(1)

        datapoints.append(tuple(measurement_list))

    BMrun["true_Zi"] = true_Zi
    BMrun["datapoints"] = datapoints
    BMrun["datapoint_labels"] = datapoint_labels

    filename = BMrun["benchmark_name"].replace(" ", "_") + '.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(BMrun, f)

    pbar.close()
    print(f"Benchmark completed: {filename}")

    return BMrun

def plot_bm_results(bm_result):
    """Plots the benchmarking results comparing VD and ZNE."""
    
    # Extract data
    X, *Ys = list(zip(*bm_result["datapoints"]))
    reference_value = vector_norm_distance(bm_result["true_Zi"], np.zeros(bm_result["N_qubits"]))
    labels = bm_result["datapoint_labels"]
    
    # Plot settings
    fig, ax = plt.subplots(figsize=(8,6))
    ax.axhline(reference_value, color='blue', lw=1, ls="--", label="Random guessing error")

    for i, Y in enumerate(Ys):
        ax.plot(X, Y, label=labels[i], marker='o', linestyle='-')

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("Expected # of Errors (Log Scale)")
    ax.set_ylabel("Square Root Distance to True State")
    ax.set_title(bm_result["benchmark_name"])
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.show()


# %%
bm_result = run_benchmarking_with_zne(
    run_name="VD_vs_ZNE_Benchmark",
    vd_iterations=[2, 4, 8],
    scale_factors=[1, 2, 3, 4],
    N_datapoints=10,
    N_qubits=6,
    N_layers=20,
    entangled=True
)

# %%
plot_bm_results(bm_result)