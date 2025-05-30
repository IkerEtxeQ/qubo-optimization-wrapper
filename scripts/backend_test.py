from qubo_optimization_wrapper.backend_handler.execution_backend import ExecutionBackend
from qubo_optimization_wrapper.hamiltonian_creation.hamiltonian import Hamiltonian


# from dwave.cloud import Client
from pyqubo import Binary, Placeholder

# Definición de variables con Binary de PyQUBO
X = [Binary(f"x_{i}") for i in range(5)]

# Definir los coeficientes de Lagrange como Placeholders
lambda_1 = Placeholder("lambda_1")
lambda_2 = Placeholder("lambda_2")

H_1R = lambda_1 * (sum(X[i] for i in range(len(X))) - 4) ** 2
H_2R = lambda_2 * ((X[0] + X[2]) - 0.5) ** 2

H = H_1R + H_2R

# Asignar los valores de las lambdas (antes de compilar)
lambda_dict = {"lambda_1": 1, "lambda_2": 1}

hamiltonian = Hamiltonian(H_1R, H_2R, lambda_dict)

hamiltonian.show_QMatrix()
# hamiltonian.show_all_hamiltonian_solutions_energy()

backend = ExecutionBackend("simulated_annealing")
backend.get_current_backend_execution_info()

backend.submit_job(hamiltonian, num_reads=10)
print(backend.get_sampleset())
