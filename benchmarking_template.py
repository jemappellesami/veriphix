import graphix.command
from graphix.random_objects import Circuit, rand_circuit
from graphix.states import BasicStates
from graphix.sim.density_matrix import DensityMatrixBackend
from graphix.sim.statevec import StatevectorBackend, Statevec
from graphix.pauli import Pauli
from graphix.fundamentals import IXYZ
import stim
from graphix.noise_models import DepolarisingNoiseModel

from veriphix.client import Client, Secrets, CircuitUtils
import veriphix.client
import numpy as np
import random
import time


nqubits = 7
depths = range(1, 10)

for depth in depths:
    print(f"## depth = {depth} ##")

    # Build the pattern
    circuit = rand_circuit(nqubits, depth)
    pattern = circuit.transpile().pattern 
    pattern.minimize_space()

    print(f"Number of nodes in the pattern : {pattern.n_node}")
    
    ## Measure output nodes, to have classical output
    classical_output = pattern.output_nodes
    for onode in classical_output:
        pattern.add(graphix.command.M(node=onode))

    # Call Veriphix
    secrets = Secrets(r=True, a=True, theta=True)
    client = Client(pattern=pattern, secrets=secrets)
    test_runs = client.create_test_runs()

    # Insert noise here !!
    noise = DepolarisingNoiseModel(entanglement_error_prob=0.01)

    backend = DensityMatrixBackend()

    n_failures = 0
    n_iterations = 5
    for _ in range(n_iterations) :
        """
        Just for benchmarking: delegating the pattern, and a test run. Execution times should not differ.
        """
        # Delegate the pattern
        t_start = time.time()
        computation_outcomes = client.delegate_pattern(backend=backend, noise_model = noise)
        t_end = time.time()
        print("C: ", t_end-t_start)

        # Delegate a random test run
        run = random.choice(test_runs)
        t_start = time.time()
        trap_outcomes = client.delegate_test_run(run=run, backend=backend, noise_model = noise)
        t_end = time.time()
        print("T: ", t_end-t_start)


        # Check trap failure (useless for benchmarking)
        if sum(trap_outcomes) != 0:
            n_failures += 1 # You can print this to display the number of failed trap rounds
            