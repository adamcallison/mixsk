import numpy as np
from numba import njit
from time import time

@njit
def cost(Jmat, hvec, state):
    return np.dot(state, np.dot(Jmat, state)) + np.dot(hvec, state)

@njit('float64(float64[:,:], float64[:,:])')
def relaxed_cost(Jmat, vecs):
    return np.trace(np.dot(Jmat, np.dot(vecs.transpose(), vecs)))

def fieldtocouplings(Jmat, hvec):
    n = Jmat.shape[0]
    Jmatlarger = np.zeros((n+1, n+1))
    Jmatlarger[:n, :n] = Jmat
    Jmatlarger[:-1,-1] = hvec
    return Jmatlarger

def symmetrise(Jmat):
    return 0.5*(Jmat + (Jmat.transpose()))

@njit('float64[:](float64[:])')
def normalise(v):
    norm = np.sqrt(np.dot(v, v))
    if norm == 0.0:
        v = np.ones(len(v))/np.sqrt(len(v))
    else:
        v = v/norm
    return v

@njit('float64[:,:](float64[:,:])')
def normalise_multi(vecs):
    nvecs = vecs.shape[1]
    newvecs = np.empty_like(vecs)
    for j in range(nvecs):
        newvecs[:,j] = normalise(vecs[:,j])
    return newvecs

def round_to_guess(vecs):
    n = vecs.shape[-1] - 1
    return np.array([np.sign(np.dot(vecs[:,-1], vecs[:,i])) for i in range(n)])

def sdp(Jmat, hvec, vecsize, max_iters=None, tol=1e-6):
    n = Jmat.shape[0]
    Jmataug = symmetrise(fieldtocouplings(Jmat, hvec))
    stepsize = 1/(1.1*np.max(
        [np.sum(np.abs(Jmataug[j])) for j in range(Jmataug.shape[0])]))
    vecs = np.random.default_rng().uniform(0.0, 1.0, (vecsize, n+1))
    vecs = normalise_multi(vecs)
    if max_iters is None:
        max_iters = -1
    vecs, loss, iters, converged = sdp_core(
        Jmataug, vecs, stepsize, max_iters, tol)
    return vecs, loss, iters, converged

@njit
def sdp_core(Jmataug, vecs, stepsize, max_iters, tol):
    n = Jmataug.shape[0]-1
    err = -1.0
    rc = relaxed_cost(Jmataug, vecs)
    losses = [rc + (200*tol), rc + (100*tol), rc]
    iters = 0
    while (max_iters == -1 or iters < max_iters) and \
        (err < 0 or (err > 0.9*tol)):
        iters += 1
        for j in range(n+1):
            newvec = np.zeros_like(vecs[:,j])
            for k in range(n+1):
                newvec += Jmataug[j, k]*vecs[:,k]
            vecs[:,j] = normalise(vecs[:,j]-(stepsize*newvec))
        losses = losses[1:] + [relaxed_cost(Jmataug, vecs)]
        err = max(
            (np.abs(losses[-1] - losses[-2]),
            np.abs(losses[-2] - losses[-3]),
            np.abs(losses[-1] - losses[-3])))
    converged = (err > 0 and (err <= 0.9*tol))
    return vecs, losses[-1], iters, converged

def bnbcost(state, Jmat, hvec):
    assert len(state) == Jmat.shape[0]
    return cost(Jmat, hvec, np.array(state, dtype=float))

def bnbbranch(state):
    lstate = list(state)
    return np.array(lstate + [1], dtype=int), np.array(lstate + [-1], dtype=int)

def bnbbound(state, Jmat, hvec, bound_type='mix', **kwargs):
    bound_type = bound_type.lower()
    if bound_type == 'mix':
        return bnbbound_mix(state, Jmat, hvec, **kwargs)
    elif bound_type == 'recursive':
        return bnbbound_recursive(state, Jmat, hvec, **kwargs)
    else:
        raise Exception

@njit
def bnbbound_recursive_core(state, Jmat, hvec, pre_computed_bounds):
    lenfixed = len(state)
    n = hvec.shape[0]
    lenfree = n - lenfixed
    Jmatfixed = Jmat[:lenfixed,:lenfixed]
    hvecfixed = hvec[:lenfixed]

    hvecfree_extra = np.zeros(lenfree)
    for j in range(lenfree):
        for k in range(lenfixed):
            hvecfree_extra[j] += \
                (Jmat[k, j+lenfixed] + Jmat[j+lenfixed, k])*state[k]

    term1 = pre_computed_bounds[lenfree-1]
    term2 = -np.sum(np.abs(hvecfree_extra))
    term3 = np.dot(state, np.dot(Jmatfixed, state))
    term4 = np.dot(hvecfixed, state)

    term = 0
    term = term + term1
    term = term + term2
    term = term + term3
    term = term + term4

    return term

def bnbbound_recursive(state, Jmat, hvec, pre_computed_bounds, **_):
    assert len(state) < Jmat.shape[0]
    lenfixed = len(state)

    if lenfixed == 0:
        return -np.float('inf'), None, 0

    term = bnbbound_recursive_core(np.array(state, dtype=float), Jmat, hvec, \
        pre_computed_bounds)

    return term, None, 0

def bnbbound_mix(state, Jmat, hvec, max_iters=None, tol=1e-8):
    assert len(state) < Jmat.shape[0]
    n = Jmat.shape[0]
    lenfixed = len(state)
    lenfree = n - lenfixed
    Jmatfixed = Jmat[:lenfixed,:lenfixed].copy()
    hvecfixed = hvec[:lenfixed].copy()
    Jmatfree = Jmat[lenfixed:,lenfixed:].copy()
    hvecfree = hvec[lenfixed:].copy()
    for j in range(lenfree):
        for k in range(lenfixed):
            hvecfree[j] += (Jmat[k, j+lenfixed] + Jmat[j+lenfixed, k])*state[k]
    fixedcost = cost(Jmatfixed, hvecfixed, state)
    if np.count_nonzero(Jmatfree) == 0 and np.count_nonzero(hvecfree) == 0:
        guess = None
        return fixedcost, guess, 0
    vecsize = int(np.ceil(np.sqrt(2*lenfree)))+2
    vecs, loss, sdp_iterations, converged = sdp(
        Jmatfree, hvecfree, vecsize, max_iters=max_iters, tol=0.1*tol)
    guess_free = np.array(np.sign(np.random.default_rng().uniform(-1.0, 1.0, \
        lenfree)), dtype=int)
    guess = np.array(list(state) + list(guess_free), dtype=int)
    if converged:
        boundval = (loss - tol) + fixedcost
    else:
        boundval = -np.float('inf')
    return boundval, guess, sdp_iterations

def bnbbound_init(Jmat, hvec, bound_type='mix', verbose=False, **kwargs):
    if bound_type == 'mix':
        return bnbbound_mix_init(Jmat, hvec, verbose=verbose, **kwargs)
    elif bound_type == 'recursive':
        return bnbbound_recursive_init(Jmat, hvec, verbose=verbose, **kwargs)
    else:
        raise Exception

def bnbbound_mix_init(Jmat, hvec, max_iters=None, tol=1e-8, verbose=False):
    return {'max_iters':max_iters, 'tol':tol}

def bnbbound_recursive_init(Jmat, hvec, pre_computed_bounds=None, \
    tol=0.0, verbose=False):
    if not pre_computed_bounds is None:
        return {'pre_computed_bounds':pre_computed_bounds, 'tol':tol}
    n = hvec.shape[0]
    pre_computed_bounds = [-np.abs(hvec[-1])]

    for ncur in range(2, n):
        #print(f'Doing {ncur}  ', end='\r')
        lenfree = ncur
        lenfixed = n - lenfree
        Jmatfree = Jmat[lenfixed:,lenfixed:].copy()
        hvecfree = hvec[lenfixed:].copy()

        tmp = bnb(Jmatfree, hvecfree, bound_type='recursive', \
            pre_computed_bounds=np.array(pre_computed_bounds), tol=tol, \
            verbose=verbose)

        new_bound = tmp[1]

        pre_computed_bounds.append(new_bound)

    return {'pre_computed_bounds':np.array(pre_computed_bounds), 'tol':tol}

def bnb(Jmat, hvec, priority_order='lowest', bound_type='mix', verbose=False, \
    **kwargs):
    # for 'mix': kwargs is 'max_iters=None' and 'tol=1e-8'
    # for 'recursive': kwargs is 'pre_computed_bounds'
    kwargs = bnbbound_init(Jmat, hvec, bound_type=bound_type, verbose=verbose, \
        **kwargs)

    tol = kwargs['tol']

    n = hvec.shape[0]
    stats = {'n':n, 'costs':0, 'bounds':0, 'branches':0, 'prunes':0, 'pruned':0,
        'sdp_max_iterations':0, 'sdp_total_iterations':0}
    stime = time()
    n = Jmat.shape[0]
    best_state, best_nrg = [], np.float('inf')
    states = [[]]
    priority = [np.float('nan')]
    while len(states) > 0:
        if verbose: print(f'{stats}    ', end='\r')
        if priority_order == 'lowest':
            state = states.pop(0)
            pri = priority.pop(0)
        elif priority_order == 'highest':
            state = states.pop()
            pri = priority.pop()
        else:
            raise ValueError('invalid priority_order')
        if len(state) == n:
            nrg = bnbcost(state, Jmat, hvec)
            stats['costs'] += 1
            if (best_nrg - tol) <= nrg and nrg <= (best_nrg + tol):
                best_nrg = min((nrg, best_nrg))
                test = False
                for x in best_state:
                    if (x == state).all():
                        test = True
                        break
                if not test:
                    best_state.append(state)
            elif nrg < (best_nrg - tol):
                best_state = [state]
                best_nrg = nrg
        else:
            bound, guess, sdp_iterations = bnbbound(
                state, Jmat, hvec, bound_type=bound_type, **kwargs)
            stats['sdp_total_iterations'] += sdp_iterations
            stats['sdp_max_iterations'] = max(
                (sdp_iterations, stats['sdp_max_iterations']))
            stats['bounds'] += 1
            if bound <= best_nrg:
                newstate1, newstate2 = bnbbranch(state)
                stats['branches'] += 1
                idx = np.searchsorted(priority, bound)
                states = states[:idx] + [newstate1, newstate2] + states[idx:]
                priority = priority[:idx] + [bound, bound] + priority[idx:]
            else:
                stats['prunes'] += 1
                stats['pruned'] += 2**(n-len(state))
            if priority_order == 'lowest' and (not (guess is None)):
                states = [guess] + states
                priority = [-np.float('inf')] + priority
            elif priority_order == 'highest' and (not (guess is None)):
                states = states + [guess]
                priority = priority + [np.float('inf')]
    stats['time'] = time() - stime
    return best_state, best_nrg, stats
