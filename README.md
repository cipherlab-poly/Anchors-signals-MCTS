# Anchors-STL-MCTS

This repository implements the algorithm introduced in our paper:
- Tzu-yi Chiu, Jérôme Le Ny, and Jean Pierre David, 
"Temporal Logic Explanations for Dynamic Decision Systems using 
Anchors and Monte Carlo Tree Search", 
*The journal of Artificial Intelligence (AIJ)*, 
[under review] 2022

## Abstract

For many automated perception and decision tasks, state-of-the-art performance 
may be obtained by algorithms that are too complex for their behavior to be 
completely understandable or predictable by human users, e.g., because they 
employ large machine learning models.
To integrate these algorithms into safety-critical decision and control systems, 
it is particularly important to develop methods that can promote trust into 
their decisions and help explore their failure modes.
**In this article, we combine the *anchors* methodology with 
*Monte Carlo Tree Search* to provide local model-agnostic explanations for 
the behaviors of a given black-box model making decisions by processing 
time-varying input signals**. 
Our approach searches for highly descriptive explanations for these decisions
in the form of properties of the input signals, expressed in Signal Temporal 
Logic, which are most susceptible to reproduce the observed behavior. 
To illustrate the methodology, we apply it in simulations to the analysis of
a hybrid (continuous-discrete) control system and a collision avoidance
system for unmanned aircraft (ACAS Xu) implemented by a neural network.

## Repository organization

```
--- main.py    - Main executable file 
 |- mcts.py    - Tree object implementing MCTS steps 
 |- stl.py     - STL objects (primitives & formulas)
 |- visual.py  - For visualization of a tree 
 |- simulators - Simulators that can generate signal samples
 |- demo       - Figures and important log files
 |_ ...
```

## Case studies

For each case study, please refer to the corresponding section of the paper 
for more context.

### Thermostat: an illustrative example (Section 4.3)

```
python3 main.py
```

### Automatic transmission system (Section 5)

```
python3 main.py
```

### ACAS Xu (Section 6)

```
python3 main.py
```
