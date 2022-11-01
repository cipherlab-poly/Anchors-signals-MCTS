This repository implements the algorithm introduced in our paper:
- Tzu-yi Chiu, Jérôme Le Ny, and Jean Pierre David, 
**"Temporal Logic Explanations for Dynamic Decision Systems using 
Anchors and Monte Carlo Tree Search"**, 
*The journal of Artificial Intelligence (AIJ)*, 
[under review] 2022

# Abstract

For many automated perception and decision tasks, state-of-the-art 
performance may be obtained by algorithms that are too complex for 
their behavior to be completely understandable or predictable by human 
users, e.g., because they employ large machine learning models.
To integrate these algorithms into safety-critical decision and control 
systems, it is particularly important to develop methods that can 
promote trust into their decisions and help explore their failure modes.
**In this article, we combine the *anchors* methodology with 
*Monte Carlo Tree Search* to provide local model-agnostic explanations 
for the behaviors of a given black-box model making decisions by 
processing time-varying input signals**. 
Our approach searches for highly descriptive explanations for these 
decisions in the form of properties of the input signals, expressed in 
*Signal Temporal Logic*, which are most susceptible to reproduce the 
observed behavior. 
To illustrate the methodology, we apply it in simulations to the 
analysis of a hybrid (continuous-discrete) control system and a 
collision avoidance system for unmanned aircraft (ACAS Xu) implemented 
by a neural network.

# Repository organization

```
 |- main.py      - Main executable script
 |- mcts.py      - Tree object implementing MCTS steps 
 |- stl.py       - STL objects (primitives & formulas)
 |- visual.py    - Script for visualization of the tree evolution (Section 4.3) 
 |- simulator.py - Abstract class for simulators
 |
 | **folders**
 |- simulators   - Simulators that can generate signal samples (function `simulate`)
 |- demo         - Figures and important log files
 |_ log          - Automatically generated log files
```

# Case studies

We implemented 3 simulators:
- Intelligent thermostat: an illustrative example, for visualization (Section 4.3)
- Automotive automatic transmission system (Section 5)
- ACAS Xu (Section 6)

For each case study, please refer to the corresponding section of the 
paper for more context.
We always assume that the simulated model is totally unknown (black-box) 
to our algorithm. 
The experiments were run on Linux with an Intel i7-7700K CPU.
The code is developed in Python 3.8 and not optimized for efficiency.

## Intelligent thermostat: explaining why switched off (Section 4.3)

Consider an automated thermostat which switches off whenever the 
detected temperature is once greater than 20 degrees Celsius within 
the past two seconds. 
Suppose that this mechanism is unknown to our algorithm but that we can 
perform as many simulations of this thermostat as we wish. 

In our scenario, the thermostat is switched off automatically, the 
decision "off" corresponding to the observed output for which we seek 
to provide an explanation.

See the function `thermostat` defined in `main.py`.

## Automotive automatic transmission system (Section 5)

### Explaining an STL-based monitoring system (Section 5.2)

To evaluate and validate the proposed algorithm, we consider five of 
the requirements on an automotive automatic transmission system:
- `G[0,10](espd<4750)` 
- `G[0,20](vspd<120)`
- `G[0,30](espd<3000) => G[0,4](vspd<35)`
- `G[0,30](espd<3000) => G[0,8](vspd<50)`
- `G[0,30](espd<3000) => G[0,20](vspd<65)`

Suppose that a monitoring system triggers an alarm when a requirement is 
violated. 
We consider each of these monitoring systems as a black-box model and 
aim to *recover* the formulas defining them.
In other words, we try to explain why the alarm has been triggered just 
by observing a signal. 

See the functions `auto_trans_alarm1` to `auto_trans_alarm5` defined in 
`simulators` and `main.py`.

### Explaining the transmission during a passing maneuver (Section 5.3)

We now focus on a scenario where the vehicle is performing a passing 
maneuver. 
Initially the vehicle is accelerating with the throttle linearly 
decreasing from 60% to 40%, up-shifting the vehicle to the 4th gear. 
At the 12th second, the throttle is suddenly pressed to 100%,
making the transmission system down-shift to the 3rd gear.
The shifting schedule of the transmission system is assumed unknown. 
We attempt to find automatically a (local) rule explaining why the 
system engaged the 3rd gear at the 12th second, by analyzing the 
throttle opening, the engine speed and the vehicle speed in the 
previous seconds, using PtSTL (Past Time STL).

See the function `auto_trans` defined in `simulators` and `main.py`.

## ACAS Xu: explaining an advisory change (Section 6)

ACAS Xu is a system implementing the decision making logic of an ACAS 
(Airborne Collision Avoidance System) specifically for unmanned aerial 
vehicles. 
It uses dynamic programming to provide maneuver guidance maintaining 
horizontal and vertical separation between two aircraft.

In our scenario, the system issued an SRT (Strong Right Turn) advisory 
for the ownship from the very beginning during 10 seconds, and switched 
to WRT (Weak Right Turn) and finally COC (Clear Of Conflict) when the 
two aircraft were no longer in danger of colliding with each other. 
We attempt to find an explanation, expressed in PtSTL, for why the 
advisory switched from SRT to WRT at the 10th second.

See the function `acas_xu` defined in `simulators` and `main.py`.

# Usage

In `main.py`, multiple case studies can be run successively by 
uncommenting the corresponding lines:
```python
def main(log_to_file: bool = False) -> None:
    "Run algorithm in multiple case studies."
    set_logger() # log to terminal
    simulators = []
    simulators.append('thermostat')
    #simulators.append('auto_trans_alarm1')
    #simulators.append('auto_trans_alarm2')
    #simulators.append('auto_trans_alarm3')
    #simulators.append('auto_trans_alarm4')
    #simulators.append('auto_trans_alarm5')
    #simulators.append('auto_trans')
    #simulators.append('acas_xu')
    for simulator in simulators:
        if log_to_file:
            set_logger(simulator)
        run(simulator)
```

The argument `--log [-l]` logs the (intermediate & final)
results to the `log` folder:
```
python3 main.py [--log [-l]]
```

For the intelligent thermostat specifically, the evolution of the tree 
(DAG) can be visualized with `visual.py`:
```
python3 visual.py
```
