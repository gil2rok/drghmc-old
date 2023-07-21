import argparse
from collections import namedtuple

import numpy as np
from sklearn.model_selection import ParameterGrid

from utils import my_save, call_counter
from samplers import bayes_kit_hmc, bayes_kit_mala, hmc, ghmc, drhmc, drghmc

HyperParamsTuple = namedtuple(
    "hyper_params",
    [
        "model_num",
        "chain_num",
        "burn_in_gradeval",
        "chain_length_gradeval",
        "global_seed",
        "save_dir",
        "pdb_dir",
        "bridgestan_dir",
    ],
)
SamplerParamsTuple = namedtuple(
    "model_params",
    [
        "init_stepsize",
        "reduction_factor",
        "steps",
        "dampening",
        "num_proposals",
        "probabilistic",
    ],
    defaults=[None, 2, 1, 0, 1, False],
)


def experiment(sampler, hp, burn_in, chain_len):
    burned_draws = np.asanyarray([sampler.sample()[0] for _ in range(burn_in)])
    
    # ignores the first call to log_density_gradient() in the sampler's init function
    # this drops the one-off gradient call needed to run leapfrog for the first time
    # because the gradient has not yet been cached
    sampler._model.log_density_gradient = call_counter(
        sampler._model.log_density_gradient
    )
    sampler._model.log_density = call_counter(sampler._model.log_density)
    draws = np.asanyarray([sampler.sample()[0] for _ in range(chain_len)])
    
    return burned_draws, draws


def bayes_kit_hmc_runner(hp):
    sampler_type = "bk_hmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "steps": [20],
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = bayes_kit_hmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)
        
        
def bayes_kit_mala_runner(hp):
    sampler_type = "bk_mala"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = bayes_kit_mala(hp, sp)

        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def hmc_runner(hp):
    sampler_type = "hmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "steps": [20],
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = hmc(hp, sp)
        
        burn_in = int(hp.burn_in_gradeval / (sp.steps))
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def ghmc_runner(hp):
    sampler_type = "ghmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "dampening": [0.01] # dampening = 0 : MALA      dampening = 1 : HMC
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = ghmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / sp.steps)
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def drhmc_runner(hp):
    sampler_type = "drhmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "reduction_factor": [2],
            "steps": [20],
            "num_proposals": [3],
            "probabilistic": [False],
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drhmc(hp, sp)

        burn_in = int(hp.burn_in_gradeval / sp.steps)
        chain_len = int(hp.chain_length_gradeval / sp.steps)

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


def drghmc_runner(hp):
    sampler_type = "drghmc"
    sampler_param_grid = ParameterGrid(
        {
            "init_stepsize": [1e-2],
            "reduction_factor": [2],
            "steps": ["const_stepsize"],
            "dampening": [1e-2],
            "num_proposals": [3],
            "probabilistic": [False],
        }
    )

    for sampler_params in sampler_param_grid:
        sp = SamplerParamsTuple(**sampler_params)
        sampler = drghmc(hp, sp)

        burn_in = (
            int(hp.burn_in_gradeval / sp.steps)
            if type(sp.steps) is int
            else hp.burn_in_gradeval
        )
        chain_len = (
            int(hp.chain_length_gradeval / sp.steps)
            if type(sp.steps) is int
            else hp.chain_length_gradeval
        )

        burned_draws, draws = experiment(sampler, hp, burn_in, chain_len)
        my_save(sp, hp, burned_draws, draws, sampler_type, sampler)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=int, help="PDB model number")
    parser.add_argument("--chain_num", type=int, help="Markov chains number")
    args = parser.parse_args()

    hp = HyperParamsTuple(
        model_num=args.model_num,
        chain_num=args.chain_num,
        burn_in_gradeval=100,  # initialize with reference sample, don't require real burn-in
        chain_length_gradeval=100000,
        global_seed=0,
        save_dir="res",
        # pdb_dir='/mnt/ceph/users/cmodi/PosteriorDB/',
        pdb_dir="models",
        bridgestan_dir="../../.bridgestan/bridgestan-2.1.1/",
    )

    # bayes_kit_mala_runner(hp)
    hmc_runner(hp)
    ghmc_runner(hp)
    drhmc_runner(hp)
    drghmc_runner(hp)
