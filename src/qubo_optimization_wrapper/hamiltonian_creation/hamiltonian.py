import dimod
from qubo_optimization_wrapper.visualization.v_hamiltonian import (
    show_QMatrix,
    visualize_energies,
)


class Hamiltonian:
    def __init__(self, natural_hamiltonian, restriction_hamiltonian, lambda_dict):
        self._natural_hamiltonian = natural_hamiltonian
        self._restriction_hamiltonian = restriction_hamiltonian
        self._hamiltonian = natural_hamiltonian + restriction_hamiltonian
        self._lambda_dict = lambda_dict
        self._compiled_hamiltonian = Compiled_Hamiltonian(
            self.get_hamiltonian(), self.get_lambda_dict()
        )

    def get_lambda_dict(self):
        return self._lambda_dict

    def get_hamiltonian(self):
        return self._hamiltonian

    def get_compiled_hamiltonian(self):
        return self._compiled_hamiltonian

    def show_QMatrix(self, eliminar_bajo_diagonal=True):
        show_QMatrix(self.get_compiled_hamiltonian(), eliminar_bajo_diagonal)

    def show_all_hamiltonian_solutions_energy(self):
        """Generatel all solutions for Hamiltonian, evaluate solutiona and get energies from bqm. Create a graph asociating energy to solution."""
        visualize_energies(
            self._lambda_dict, self._hamiltonian, self._compiled_hamiltonian
        )


class Compiled_Hamiltonian:
    def __init__(self, H, lambda_dict=None):
        """
        Inicializa y compila el Hamiltoniano.

        Args:
            H: La expresión simbólica del Hamiltoniano de PyQUBO. Debe ser una instancia de pyqubo.Expression.
            lambda_dict (dict, optional): Diccionario que mapea los Placeholder de PyQUBO
                                         a sus valores numéricos. Por defecto es None.

        Raises:
            TypeError: Si H no es una expresión de PyQUBO o lambda_dict no es un diccionario (si se proporciona).
            AttributeError: Si H no tiene un método 'compile' (sugiere que no es una expresión de PyQUBO).
        """
        self._model = None
        self._bqm = None

        if lambda_dict is None:
            lambda_dict = {}
        elif not isinstance(lambda_dict, dict):
            raise TypeError(
                f"lambda_dict debe ser un diccionario o None, pero se obtuvo {type(lambda_dict)}"
            )
        else:
            self.lambda_dict = lambda_dict

        self.compilar_hamiltoniano(H, lambda_dict)

    def get_model(self):
        """Retorna el modelo compilado de PyQUBO."""
        return self._model

    def get_bqm(self):
        """Retorna el modelo BQM (BinaryQuadraticModel) de dimod."""
        return self._bqm

    def compilar_hamiltoniano(self, H, lambda_dict):
        """
        Compila el Hamiltoniano de PyQUBO y lo convierte a un modelo BQM.

        Args:
            H: El Hamiltoniano de PyQUBO (expresión simbólica). Debe ser una instancia de pyqubo.Expression.
            lambda_dict (dict): Diccionario que mapea los Placeholder a sus valores numéricos.

        Raises:
            TypeError: Si H no es una expresión de PyQUBO.
            AttributeError: Si H no tiene un método 'compile'.
        """

        try:
            self._model = H.compile()
        except AttributeError as e:
            raise AttributeError(
                f"El objeto H no tiene un método 'compile'. Asegúrate de que es una expresión de PyQUBO válida. Error original: {e}"
            )
        except Exception as e:
            raise RuntimeError(f"Error al compilar el Hamiltoniano H: {e}")

        try:
            if lambda_dict:
                self._bqm = self._model.to_bqm(feed_dict=lambda_dict)
            else:
                self._bqm = self._model.to_bqm()
        except Exception as e:
            raise RuntimeError(f"Error al convertir el modelo a BQM: {e}")

        if not isinstance(self._bqm, dimod.BinaryQuadraticModel):
            raise TypeError(
                f"Se esperaba que self._model.to_bqm() retornara un dimod.BinaryQuadraticModel, pero se obtuvo {type(self._bqm)}"
            )
