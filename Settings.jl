
settings_file = "none" #lim0.0875_mcts"#mpc_unk_reg_depth10" # name of data file to load
settings_folder = "" #set4" # store data files here

# SIM SETTINGS
prob = "2D" # set to the "1D" or "2D" problems defined
sim = "mpc" # mcts, mpc, qmdp, smpc, snmpc
rollout = "mpc3" # MCTS/QMDP: random/position/mpc/valest
lin_params = false # if want to estimate 1/m and 1/J (not directly m or J) then set to true
trial_parallel = false # parallelize by num_trials for non-CE runs
state_mean = false # sample mean or rand of the state during transition in MDP
bounds = false # set bounds for mcts solver
bounds_print = false # print results for bounds
bounds_save = false # save file with bounds trial data
desired_bounds = 6.0 # norm(1.2*ones(ssm.nx,1)) # setting for the limit to the ||Xt+1|| (maybe make in addition to the previous state?)
quick_run = true
numtrials = 5 # number of simulation runs
noiseList = []
cond1 = "full"
trials_offset = 2 # offset of number of trials to start count and correctly initialize params.

#NOISE SETTINGS
processNoiseList = [0.033]#[0.033, 0.1]#[0.001,0.0033,0.01,0.033,0.1,0.33] # default to full
paramNoiseList = [0.5]#,0.5,0.7]
ukf_flag = true # use ukf as the update method when computing mcts predictions
param_change = false # add a cosine term to the unknown param updates
param_type = "none" # sine or steps
param_magn = 0.2 # magnitude of cosine additive term # use >0.6 for steps
param_freq = 0.3
add_noise = 0.0

# Output settings
printing = false # set to true to print simple information
print_iters = false
print_trials = false
plotting = false # set to true to output plots of the data
saving = false # set to true to save simulation data to a folder # MCTS trial at ~500 iters is 6 min ea, 1hr for 10
tree_vis = false # visual MCTS tree
sim_save = "testing" # name appended to sim settings for simulation folder to store data from runs
data_folder = "test"
fullobs = true # set to false for mpc without full obs
if sim != "mpc" # set fullobs false for any other sim
  fullobs = false
end

# CROSS ENTROPY SETTINGS
cross_entropy = false
save_last = false # save last generation of CE trials
save_best = false # save best overall run, just the reward and std, and params info
num_pop = 8 #  number of samples to test this round of CE
num_elite = 8 # number of elite samples to keep to form next distribution
CE_iters = 3 # number of iterations for cross entropy
CE_params = 4 # number of params being sampled
states_m = 10.0
states_std = 0.1
act_m = 4.0
act_std = 0.1
depth_m = 3.0
depth_std = 0.1
expl_m = 1.0
expl_std = 0.1
max_eig_cutoff = 5.0
#global save_best_mean = -100000.0
#global save_best_std = 0.0

# Reward type settings
reward_type = "region" # L1 (standard L1 cost function) or region (for being within a desired zone)
reward_region = "small"

# Settings for simulation
measNoise = 0.000001 # standard deviation of measurement noise
deltaT = 0.1 # timestep for simulation --> decrease for complex systems?
debug_bounds = false # set to 1 to print out commands to see when limits of state and estimation hit
cov_thresh = 1000 # threshold where trace(cov) of estimate will be discarded in MCTS
state_init = 1.0 # gain for the initial state
state_min_tol = 0.1 # prevent states from growing less than X% of original value
friction_lim = 3.0 # limit to 2D friction case to prevent exploding growth
fRange = 5.0 # bounds on controls allowed within +- fRange
if quick_run
  nSamples = 5 # quick amount of steps for debug_bounds
else
  nSamples = 50
end

# settings for mcts
n_iters = 100#3000#00 # total number of iterations
samples_per_state = 2#3 # want to be small
samples_per_act = 4# want this to be ~20
depths = 20# depth of tree
expl_constant = 1.0#100.0 #exploration const

include("ReadSettings.jl") # read in new values from data file if given