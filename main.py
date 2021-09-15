import numpy as np
np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

from mcts import MCTS
from stl import STL, PrimitiveGenerator, Simulator

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-5s %(message)s',
                    datefmt='%H:%M:%S')

def simulate_thermostat(params) -> Simulator:
    from models.thermostat import Thermostat

    slen = 5
    tm = Thermostat(out_temp=19, exp_temp=20, latency=2, length=slen)
    tm.temps = [19.53, 19.33, 19.83, 20.08, 19.37]
    tm.on = 0

    params['s']     = np.array([tm.temps])
    params['range'] = [(0, (19, 21, 20))]
    params['y']     = tm.on
    return tm

def simulate_acas_xu(params) -> Simulator:
    from models.acas_xu import ACAS_XU

    state0 = np.array([5000.0, np.pi/4, -np.pi/2, 300.0, 100.0])
    acasxu = ACAS_XU(state0, tdelta=1.0, slen=10)
    acasxu.load_nnets()
    smins = [0.0, 0.0, -np.pi]
    smaxes = [8000.0, np.pi, 0.0]
    params['s'] = acasxu.run()
    params['range'] = [(0, (smins[i], smaxes[i], 8)) for i in range(3)]
    params['y'] = acasxu.a_actual
    params['tau'] = 0.98
    params['epsilon'] = 0.015
    params['past'] = True
    return acasxu

def simulate_auto_transmission(params) -> Simulator:
    from models.auto_transmission import AutoTransmission

    duration = 12
    tdelta = 0.5
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta)))
    throttles += [1.0]
    thetas = [0.] * len(throttles)
    at = AutoTransmission(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 200, 8)), (0, (0, 1, 10)), (1, [1, 2, 3, 4])]
    params['epsilon'] = 0.015
    params['tau'] = 0.98
    params['y'] = at.gear
    params['past'] = True
    return at

def simulate_auto_transmission2(params) -> Simulator:
    from models.auto_transmission2 import AutoTransmission2

    duration = 12
    tdelta = 0.5
    throttles = list(np.linspace(0.6, 0.4, int(duration/tdelta)))
    throttles += [1.0]
    thetas = [0.] * len(throttles)
    at = AutoTransmission2(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 200, 20)), (0, (0, 1, 10)), (1, [1, 2, 3, 4])]
    params['epsilon'] = 0.015
    params['tau'] = 0.98
    params['y'] = at.gear
    params['past'] = True
    return at

"""
Should be defined in params
---------------------------
s: np.ndarray
    signal being explained
range: list of tuples
    srange[d] = | (0, (min, max, stepsize))     if continuous
                | (1, list of the finite set)   if discrete
y: int
    output = black_box(s)

Optional
--------
batch_size: int
    number of samples drawn at each rollout
tau: float 
    precision threshold
rho: float
    robustness degree (~coverage) threshold
max_depth: int
    maximum depth to expand the tree
delta: float
    confidence threshold
epsilon: float
    maximum tolerated error 
"""

def main(params={}):
    #simulator = simulate_thermostat(params)
    #simulator = simulate_acas_xu(params)
    #simulator = simulate_auto_transmission(params)
    simulator = simulate_auto_transmission2(params)

    if not {'s', 'range'}.issubset(params.keys()):
        logging.error('something undefined in params among {s, range}')
        return
    s           = params.get('s',           None)
    srange      = params.get('range',       None)
    y           = params.get('y',           None)
    batch_size  = params.get('batch_size',  16  )
    tau         = params.get('tau',         0.95)
    rho         = params.get('rho',         0.01)
    max_depth   = params.get('max_depth',   6   )
    delta       = params.get('delta',       0.01)
    epsilon     = params.get('epsilon',     0.01)
    past        = params.get('past',        False)
    
    logging.info(f'Signal and output to explain:\n{s} => {y}')
    tree = MCTS(batch_size=batch_size, max_depth=max_depth, 
                delta=delta, epsilon=epsilon)
    stl = STL()
    primitives = PrimitiveGenerator(s, srange, rho, past).generate()
    logging.info('Initializing primitives...')
    nb = stl.init(primitives, simulator)
    logging.info(f'Done. {nb} primitives.')
    
    while True:
        logging.info('Choosing best primitive...')
        try:
            tree.train(stl)
        except KeyboardInterrupt:
            logging.warning('Interrupted')
        finally:
            stls = tree.choose(stl, nb=5)
            for stl in stls:
                q, n = tree.Q[stl], tree.N[stl]
                if not n:
                    logging.info(f'{stl} ({q}/{n})')
                else:
                    logging.info(f'{stl} ({q}/{n}={q/n:5.2%})')
            stl = stls[0]
            if tree.Q[stl]/tree.N[stl] > tau or len(stl) >= max_depth:
                break

    #tree.visualize()

if __name__ == '__main__':
    main()
