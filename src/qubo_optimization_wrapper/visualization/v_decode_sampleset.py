def min_energy_result(decoded_samples):
    print("RESULTADOS SIMULATED ANNEALING:")
    print("-------------------------")
    best_sample = min(decoded_samples, key=lambda d: d.energy)
    print(best_sample)
    print("")

    def decode_sampleset(hamiltonian, sampleset):
        """More info: https://test-projecttemplate-dimod.readthedocs.io/en/latest/reference/sampleset.html#id1

        Returns:
            de
        """
        model = hamiltonian.get_compiled_hamiltonian().get_model()
        lambda_dict = hamiltonian.get_lambda_dict()

        if lambda_dict:
            decoded_samples = model.decode_sampleset(sampleset, feed_dict=lambda_dict)
        else:
            decoded_samples = model.decode_sampleset(sampleset)

        return decoded_samples
