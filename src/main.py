import subprocess
import os
import json
import argparse
from pathlib import Path
from collections import namedtuple

import numpy as np
import bridgestan as bs
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tqdm import tqdm
from sklearn.model_selection import ParameterGrid

from posteriordb import BSDB
from utils import DualAveragingStepSize, my_save

# install bayes-kit locally
subprocess.run(["pip", "install", "-q", "-e", "../bayes-kit"])
from bayes_kit.drghmc import DrGhmcDiag
from bayes_kit.hmc import HMCDiag

HyperParamsTuple = namedtuple("hyper_params", [
    "model_num",
    "chain_num", 
    "burn_in", 
    "chain_length", 
    "global_seed", 
    "save_dir",
    "pdb_dir", 
    "bridgestan_dir"
])
SamplerParamsTuple = namedtuple("model_params", [
    "init_stepsize", 
    "adaptivity_factor", 
    "leapfrog_steps", 
    "dampening", 
    "num_proposals", 
    "probabilistic"
])


def get_model(hp):
    return BSDB(hp.model_num, hp.pdb_dir)


def get_init_stepsize(hp):
    model = get_model(hp)
    init_stepsize = hp.first_stepsize
    dual_avg = DualAveragingStepSize(initial_step_size=init_stepsize, nadapt=hp.burn_in)
    for i in range(hp.burn_in):
        hmc = HMCDiag(model=model, steps=10, stepsize=init_stepsize)
        _, _, acceptp = hmc.sample(return_acceptp=True)
        init_stepsize = dual_avg(i=i, p_accept=acceptp)
    return init_stepsize


def evaluate(draws, sp, hp):
    model = get_model(hp)
    fname = os.path.join(hp.pdb_dir,  f'PDB_{hp.model_num:02d}', f'PDB_{hp.model_num:02d}.samples.npy') 
    constrained = np.load(fname)[hp.chain_num, :, :].copy(order='C') # [chain_num, n_samples, params_dim]
    reference_draws = model.unconstrain(constrained)
    
    subplot_titles = [None] * (2 * model.dims())
    subplot_titles[0] = "Reference Draws"
    subplot_titles[1] = "DrGhmc Draws"
    fig = make_subplots(rows=model.dims(), cols=2, shared_yaxes=True,
                        subplot_titles=subplot_titles,)
    
        
    for i in range(model.dims()):
        cur_ref, cur_draws = reference_draws[:, i], draws[:, i]
        
        fig.add_trace(go.Histogram(
            x=cur_ref, name=f"Reference {i+1}", 
            opacity=0.5, histnorm="probability",
        ), row=i+1, col=1)
        fig.add_annotation(
            x=np.mean(cur_ref), y=0.1, 
            text=f"μ={np.mean(cur_ref):.2f}, σ={np.std(cur_ref):.2f}",
            showarrow=False, row=i+1, col=1,)
        
        fig.add_trace(go.Histogram(
            x=draws[:, i], name=f"DrGhmc {i+1}", 
            opacity=0.5, histnorm="probability",
        ), row=i+1, col=2)
        fig.add_annotation(
            x=np.mean(cur_draws), y=0.1, 
            text=f"μ={np.mean(cur_draws):.2f}, σ={np.std(cur_draws):.2f}",
            showarrow=False, row=i+1, col=2,)
        
    sp_text = "\t".join(f"{str(k)}: {str(v)}" for k, v in sp._asdict().items())
    fig.add_annotation(xref="paper", yref="paper", text=sp_text, showarrow=False, x=0, y=-0.05) # x = 1.2, y = 0.7
    
    hp_text = "\t".join(f"{str(k)}: {str(v)}" for k, v in hp._asdict().items())
    fig.add_annotation(xref="paper", yref="paper", text=hp_text, showarrow=False, x=0, y=-0.08)
        
    fig.update_layout(height=200*model.dims(), width=1000, title_text=f"Model {hp.model_num} Chain {hp.chain_num}")
    return fig


def sampler(sp, hp):
    model = get_model(hp)
    init_stepsize =  sp.init_stepsize
    stepsize = lambda k: init_stepsize * (sp.adaptivity_factor ** -k)
    steps = 1 if sp.leapfrog_steps == "const_num_steps" else lambda k: int(1 / stepsize(k))
    seed = int(str(hp.global_seed) + str(hp.chain_num)) # seed depends on global seed and chain number
    
    # remove after testing
    fname = os.path.join(hp.pdb_dir,  f'PDB_{hp.model_num:02d}', f'PDB_{hp.model_num:02d}.samples.npy') 
    init_constrained = np.load(fname)[0,-1, :].copy(order='C')
    init = model.unconstrain(init_constrained)
    
    sampler = DrGhmcDiag(
        model=model,
        stepsize=stepsize,
        steps=steps,
        seed=seed,
        init=init,
        num_proposals=sp.num_proposals,
        probabilistic=sp.probabilistic,
        dampening=sp.dampening,
    )
    
    burned_draws = np.asanyarray([sampler.sample()[0] for _ in range(hp.burn_in)])
    draws = np.asanyarray([sampler.sample()[0] for _ in range(hp.chain_length)])
    return burned_draws, draws


def experiment(sp, hp):
    burned_draws, draws = sampler(sp, hp)
    fig = evaluate(draws, sp, hp)
    my_save(sp, hp, burned_draws, draws, fig)
    return burned_draws, draws


def experiment_runner(model_num, chain_num):
    # although burn_in and chain_length are sampler parameters, 
    # we do not grid search over them and treat them as hyperparameters
    hp = HyperParamsTuple(
        model_num=model_num,
        chain_num=chain_num,
        burn_in=10, # initialize with reference sample, don't require real burn-in
        chain_length=5000,
        global_seed=0,
        save_dir='res',
        # pdb_dir='/mnt/ceph/users/cmodi/PosteriorDB/',
        pdb_dir = '.',
        bridgestan_dir='../../.bridgestan/bridgestan-2.1.1/',
    )
    
    sampler_param_grid = ParameterGrid(
        {"init_stepsize": [1e-3],
        "adaptivity_factor": [2],
        "leapfrog_steps": ["const_num_steps", "const_trajectory_length"],
        "dampening": [0.05],
        "num_proposals": [3],
        "probabilistic": [True, False]}
    )

    # bs.set_bridgestan_path(hp.bridgestan_dir)
    for sampler_params in tqdm(sampler_param_grid):
        sp = SamplerParamsTuple(**sampler_params)
        experiment(sp, hp)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_num", type=int, help="PDB model number")
    parser.add_argument("--chain_num", type=int, help="Markov chain number to run in parallel")
  
    args = parser.parse_args()
    experiment_runner(args.model_num, args.chain_num)
    