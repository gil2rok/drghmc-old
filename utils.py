from pathlib import Path
import numpy as np
import json
import os

class DualAveragingStepSize():
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt
        
    def update(self, p_accept):

        if np.isnan(p_accept) : p_accept = 0.
        if p_accept > 1: p_accept = 1. 
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n"%step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size

def compute_hash(sp, hp):
    return str(hash(str(hash(sp)) + str(hash(hp))))

def my_save(sp, hp, burned_draws, draws):
    param_hash = compute_hash(sp, hp)
    dir_name = os.path.join(hp.save_dir, hp.model_name, param_hash)
    Path(dir_name).mkdir(parents=True, exist_ok=True)
    
    np.savez_compressed(os.path.join(dir_name, 'burned_draws'), burned_draws=burned_draws)
    np.savez_compressed(os.path.join(dir_name, 'draws'), draws=draws)
    
    with open(os.path.join(dir_name, 'hyper_params.json'), 'w') as file:
        file.write(json.dumps(hp._asdict()))
    
    with open(os.path.join(dir_name, 'sampler_params.json'), 'w') as file:
        file.write(json.dumps(sp._asdict()))