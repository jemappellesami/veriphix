
- circuits pipline parameters: sweep, N_shots, BQP error
  - generate random circuits
  - sample BQP circuits
- run benchmark inspired by @applications/benchmarking.py
  - define a new parameter sweep over depolarizing noise with entanglement p_err from 1e-6 to 1e-1
  - for each parameter value of width, depth: 
    - run tests under that noise model
    - compute test rounds failure rate
- record relevant information in a CSV file