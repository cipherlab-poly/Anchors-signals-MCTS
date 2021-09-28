import numpy as np
np.set_printoptions(precision=2, suppress=True)
np.random.seed(42)

from mcts import MCTS
from kl_lucb import KL_LUCB
from stl import STL, PrimitiveGenerator, Simulator

import logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s %(levelname)-5s %(message)s',
                    datefmt='%H:%M:%S')

def simulate_thermostat(params) -> Simulator:
    from models.thermostat import Thermostat
    
    tm = Thermostat(out_temp=19, exp_temp=20, latency=2, length=5)
    params['s']     = np.array([[19.53, 19.33, 19.83, 20.08, 19.37]])
    params['range'] = [(0, (19, 21, 20))]
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
    params['past'] = True
    return at

def simulate_auto_transmission3(params) -> Simulator:
    from models.auto_transmission3 import AutoTransmission3
    
    tdelta = 1.0
    throttles = [0.55]*5 + [0.9]*5 + [0.55]*5
    thetas = [0.]*len(throttles)

    at = AutoTransmission3(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (1000, 6000, 20)), (0, (0, 160, 8))]
    params['epsilon'] = 0.02
    params['tau'] = 0.99
    params['batch_size'] = 8
    return at

def simulate_auto_transmission4(params) -> Simulator:
    from models.auto_transmission4 import AutoTransmission4
    
    tdelta = 1.0
    throttles = [0.9] * 21
    thetas = [0.]*len(throttles)

    at = AutoTransmission4(throttles, thetas, tdelta)
    params['s'] = at.run()
    params['range'] = [(0, (0, 6000, 6)), (0, (0, 140, 7))]
    params['epsilon'] = 0.02
    params['rho'] = 0.05
    params['max_depth'] = 3
    params['tau'] = 0.99
    params['batch_size'] = 16
    return at

"""
Should be defined in params
---------------------------
s: np.ndarray
    signal being explained
range: list of tuples
    srange[d] = | (0, (min, max, stepsize))     if continuous
                | (1, list of the finite set)   if discrete

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
    #simulator = simulate_auto_transmission2(params)
    #simulator = simulate_auto_transmission3(params)
    simulator = simulate_auto_transmission4(params)

    method = 'MCTS'
    #method = 'KL-LUCB'

    if not {'s', 'range'}.issubset(params.keys()):
        logging.error('something undefined in params among {s, range}')
        return
    s           = params.get('s',           None)
    srange      = params.get('range',       None)
    batch_size  = params.get('batch_size',  16  )
    tau         = params.get('tau',         0.95)
    rho         = params.get('rho',         0.01)
    epsilon     = params.get('epsilon',     0.01)
    past        = params.get('past',        False)

    logging.info(f'Signal being analyzed:\n{s}')
    stl = STL()
    primitives = PrimitiveGenerator(s, srange, rho, past).generate()
    logging.info('Initializing primitives...')
    nb = stl.init(primitives, simulator)
    logging.info(f'Done. {nb} primitives.')

    interrupted = False
    if method == 'MCTS':
        max_depth = params.get('max_depth', 4)
        tree = MCTS(batch_size=batch_size, max_depth=max_depth, 
                    epsilon=epsilon)
        while not interrupted:
            logging.info('Choosing best primitive...')
            try:
                tree.train(stl)
            except KeyboardInterrupt:
                logging.warning('Interrupted')
                interrupted = True
            finally:
                stl = tree.choose(stl)
                q, n = tree.Q[stl], tree.N[stl]
                if not n:
                    logging.info(f'{stl} ({q}/{n})')
                    return
                logging.info(f'{stl} ({q}/{n}={q/n:5.2%})')
                if q / n > tau or len(stl) >= max_depth:
                    return
    else:
        beam_width = params.get('beam_width', 1)
        delta = params.get('delta', 0.01)
        tree = KL_LUCB(batch_size=batch_size, beam_width=beam_width, 
                        delta=delta, epsilon=epsilon)
        cands = {stl}
        while not interrupted:
            logging.info('Choosing best primitive...')
            cands = tree.get_cands(cands)
            try:
                tree.train(cands)
            except KeyboardInterrupt:
                logging.warning('Interrupted')
                interrupted = True
            stls = tree.choose(cands)
            for stl in stls:
                q, n = tree.Q[stl], tree.N[stl]
                lb, ub = tree.LB[stl], tree.UB[stl]
                logging.info(f'{stl} [{lb:5.2%}, {q}/{n}={q/n:5.2%}, {ub:5.2%}]')
            stl = stls[0]
            if tree.Q[stl] / tree.N[stl] > tau or len(stl) >= max_depth:
                return
            else:
                cands = stls

if __name__ == '__main__':
    main()
