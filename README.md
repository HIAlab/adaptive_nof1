[![Lint](https://github.com/HIAlab/adaptive_nof1/actions/workflows/lint.yml/badge.svg)](https://github.com/HIAlab/adaptive_nof1/actions/workflows/lint.yml)

# Adaptive N-of-1 trials
This repo contains code and data for Dominik's master thesis on adaptive N-of-1 trials

# Design Rationales
This framework follows some rationales which make it easier to join the projects:
- Supposed split between "Runners" and "Data", e.g. SimulationRunner and SimulationData
Runners are used to create SimulationData. This explicit split is useful, when Simulation starts to take a long time, for example cause to training of Bayesian Models in each timestep. The SimulationData created is supposed to contain all necessary information.
This split also makes it possible to execute the runner on e.g., a cluster, and then analyze the SimulationData locally.

- Extensive use of composite pattern for policies
