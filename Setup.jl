
# SIM SETTINGS
prob = "2D" # set to the "1D" or "2D" problems defined
sim = "mpc" # mcts, mpc, qmdp, drqn
rollout = "random" # MCTS/QMDP: random/position, DRQN: train/test
state_mean = true # sample mean or rand of the state during transition in MDP
bounds = false # set bounds for mcts solver
bounds_print = false # print results for bounds
quick_run = false
numtrials = 50 # number of simulation runs
noiseList = []
cond1 = "full"

#NOISE SETTINGS
processNoiseList = [0.033]#[0.033, 0.1]#[0.001,0.0033,0.01,0.033,0.1,0.33] # default to full
paramNoiseList = [0.5]#,0.5,0.7]
ukf_flag = true # use ukf as the update method when computing mcts predictions
param_change = false # add a cosine term to the unknown param updates
param_type = "none" # sine or steps
param_magn = 0.2 # magnitude of cosine additive term # use >0.6 for steps
param_freq = 0.3

# Output settings
printing = false # set to true to print simple information
print_iters = false
plotting = false # set to true to output plots of the data
saving = false # set to true to save simulation data to a folder # MCTS trial at ~500 iters is 6 min ea, 1hr for 10
tree_vis = false # visual MCTS tree
sim_save = "MPC_fobs_0.033_0.5" # name appended to sim settings for simulation folder to store data from runs
data_folder = "MPC"
fullobs = true # set to false for mpc without full obs
if sim != "mpc" # set fullobs false for any other sim
  fullobs = false
end

# CROSS ENTROPY SETTINGS
cross_entropy = false
save_last = false # save last generation of CE trials
save_best = false # save best overall run, just the reward and std, and params info
num_pop = 6 #  number of samples to test this round of CE
num_elite = 6 # number of elite samples to keep to form next distribution
CE_iters = 10 # number of iterations for cross entropy
CE_params = 3 # number of params being sampled
states_m = 10.0
states_std = 0.1
act_m = 1.0
act_std = 0.1
depth_m = 3.0
depth_std = 0.1
expl_m = 1.0
expl_std = 0.1
max_eig_cutoff = 1.0
#global save_best_mean = -100000.0
#global save_best_std = 0.0

# Reward type settings
reward_type = "L1" # L1 (standard L1 cost function) or region (for being within a desired zone)
if prob == "2D"
    state_bound_value = 0.5#0.05
    th_bound_value = 1.0#0.174
    region_lb = [-state_bound_value, -state_bound_value, -th_bound_value] # values for x,y,theta lower bound for being in the reward area
    region_ub = [state_bound_value, state_bound_value, th_bound_value]
    rew_in_region = 0.0
    rew_out_region = -1.0
end

# Settings for simulation
measNoise = 0.000001 # standard deviation of measurement noise
deltaT = 0.1 # timestep for simulation --> decrease for complex systems?
debug_bounds = false # set to 1 to print out commands to see when limits of state and estimation hit
cov_thresh = 1000 # threshold where trace(cov) of estimate will be discarded in MCTS
state_init = 1.0 # gain for the initial state
state_min_tol = 0.1 # prevent states from growing less than X% of original value
friction_lim = 3.0 # limit to 2D friction case to prevent exploding growth

# settings for mcts
n_iters = 10000#3000#00 # total number of iterations
samples_per_state = 1#3 # want to be small
samples_per_act = 40 # want this to be ~20
depths = 20 # depth of tree
expl_constant = 0.1#100.0 #exploration const

include("ReadSettings.jl") # read in new values from data file if given

if quick_run
  nSamples = 5 # quick amount of steps for debug_bounds
else
  nSamples = 50
end

if prob == "2D"
  # Settings for simulation
  fRange = 5.0 # bounds on controls allowed within +- fRange
  state_init = 1.0 # gain for the initial state
  fDist_disc = 1000 # discrete points in fDist force linspace
  # Reward shaping
  Qg = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # for use in POMDP reward funct
  Qr = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0]) # for computing reward below just for measured states
  Rg = 0.25*diagm([10.0, 10.0, 10.0])

  if (sim == "mcts") || (sim == "qmdp")
    # Parameters for the POMDP
    #500 = 7s, 2000 = 30s, 5000 = 60s
    alpha_act = 1.0/30.0 # alpha for action
    alpha_st = 1.0/30.0 # alpha for state
    k_act = samples_per_act/(n_iters^alpha_act) # k for action
    k_st = samples_per_state/(n_iters^alpha_st) # k for state
    pos_control_gain = -80.0 # gain to drive position rollout --> higher = more aggressive
    #control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "mpc"
    n = 50 # horizon steps
  end
end

# Packages
if (sim == "mcts") || (sim == "qmdp")
  using Distributions, POMDPs, MCTS, POMDPToolbox # for MCTS
    if tree_vis
      using D3Trees
    end
  if !ukf_flag
      using ForwardDiff
  end
elseif sim == "mpc"
  using Distributions, Convex, SCS, ECOS# for MPC
  if fullobs
    rollout = "fobs"
  else
    rollout = "unk"
  end
end
#using ForwardDiff # for EKF
if saving || save_last
  include("Save.jl")
end
if plotting
  using Plots
  plotly()#js()
end

# First have to load SSM to define params for rest of the setup
include("SSM.jl") # contains SSM definitions and functions
if prob == "2D"
  ssm = build2DSSM(deltaT)#,processNoise,measNoise) # building state-space
  prob_params = ["vx","vy","w","x","y","theta","m","uv","J","rx","ry"]
  control_params = ["Fx", "Fy", "T"]
end

# Force limits in the problem --> need before defining MPC/POMDP
FVar = fRange;
TVar = fRange;
fDist = linspace(-fRange, fRange, fDist_disc)
uDist = MvNormal(zeros(ssm.nu),diagm(fRange*ones(ssm.nu)))
startState = Int(ssm.states/2+1) # first position index
if startState == ssm.states
  pos_range = startState
else
  pos_range = startState:ssm.states # range of indeces in state mean for positions used in propotional controller
end

# initialize first process and param noises and Q,R
processNoise = processNoiseList[1]
paramNoise = paramNoiseList[1]
Q = diagm(processNoise*ones(ssm.nx))
R = diagm(measNoise*ones(ssm.ny))

# Loading scripts based on simulation and problem conditions
if !ukf_flag
    include("EKF.jl") # contains EKF
else
    include("UKF.jl") # contains UKF
end
if prob == "2D" # load files for 2D problem
  include("LimitChecks_2D.jl") # checks for control bounds and state/est values
  if sim == "mcts"
    include("POMDP_2D.jl") # functions for POMDP definition
  elseif sim == "qmdp"
    include("QMDP_2D.jl")
  elseif sim == "mpc"
    include("MPC_2D.jl") # function to set up MPC opt and solve
  end
end


if bounds || bounds_print
  clip_bounds = true
  if prob == "2D"
      state_init = 1.0 # gain for the initial state
      state_min_tol = 0.1 # prevent states from growing less than X% of original value
      friction_lim = 3.0
      ub_clip_lim = [Inf*ones(6); state_init*state_min_tol; friction_lim; state_init*state_min_tol*ones(3)] # only upper bound friction to prevent explosion
      lb_clip_lim = [-Inf*ones(6); state_init*state_min_tol*ones(5)]
  end
  include("EllipseBounds.jl")
  # will need to precompute this for each ProcessNoise case in the sim file so add it to main simulation.jl soon #TODO
  @show desired_bounds = 6.0#norm(1.2*ones(ssm.nx,1)) # setting for the limit to the ||Xt+1|| (maybe make in addition to the previous state?)
  n_w = 100 #30
  n_out_w = 100 #10
  F = 1.93 #2.34
  w_bound_samples = ellipsoid_bounds(MvNormal(zeros(ssm.nx),processNoiseList[1]*eye(ssm.nx,ssm.nx)),n_w,n_out_w,F) # precompute the w_bound
  w_bound_avg = mean(w_bound_samples,2) # average samples of the w_bound
  @show w_bound = norm(w_bound_avg*sqrt(processNoiseList[1])) # compute the norm of the STD * the w_bound
  max_action_count = 10 # how many actions to check before giving up on finding a feasible one
  action_count = 1
end

if tree_vis
  hist = HistoryRecorder() # necessary for POMDP setup #zach: is there a way to extract all actions and rewards for each step?
end

# solver defined here with all settings
if (sim == "mcts") || (sim == "qmdp")
  if rollout == "random" && bounds == true
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll))#, enable_tree_vis = tree_vis)#-4 before
  elseif rollout == "random"
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)#, enable_tree_vis = tree_vis)#-4 before
  end
  policy = solve(solver,mdp) # policy setup for POMDP
end

# save name and if CE save file else setup noiseList
sim_save_name = string(sim_save,"_",prob,"_",sim,"_",cond1,"_",param_type,"_",fullobs)
#@show sim_save_name

# pass in current MvNormal and get out clipped samples
function CE_sample(distrib::MvNormal,num_samples::Int,iters::Int,process::Float64,param::Float64,name::String,CE_count::Int)
    output = []
    for i in 1:num_samples
        temp_CE = rand(distrib)
        # clip samples so the first 3 are min 1, and last is min 0.1

        temp_CE[1] = max(1.0,round(temp_CE[1]))
        temp_CE[2] = max(1.0,round(temp_CE[2]))
        temp_CE[3] = max(0.1,round(temp_CE[3]))

        #temp_CE[4] = max(0.1,floor(temp_CE[4]))

        #push!(output,(temp_CE[1],temp_CE[2],temp_CE[3],temp_CE[4],iters,process,param,name,i,CE_count))
        push!(output,(samples_per_state,temp_CE[1],temp_CE[2],temp_CE[3],iters,process,param,name,i,CE_count))
    end
    return output
end

if cross_entropy
    #CEset_list = [states_m,states_std,act_m,act_std,depth_m,depth_std,expl_m,expl_std]
    CEset_list = [act_m,act_std,depth_m,depth_std,expl_m,expl_std]

    CEset = MvNormal([act_m,depth_m,expl_m],diagm([act_std^2,depth_std^2,expl_std^2]))
    distrib = CEset
    #=
    pmapInput = []
    for i in 1:num_pop
        #push!(pmapInput,(rand(CEset[1]:CEset[2]),rand(CEset[3]:CEset[4]),rand(CEset[5]:CEset[6]),rand(CEset[7]:CEset[8]),rand(CEset[9]:CEset[10]),processNoiseList[1],paramNoiseList[1],sim_save_name,i,1))
        temp_CE = rand(CEset)
        push!(pmapInput,(temp_CE[1],temp_CE[2],temp_CE[3],temp_CE[4],n_iters,processNoiseList[1],paramNoiseList[1],sim_save_name,i,1))
    end
    =#
    pmapInput = CE_sample(distrib,num_pop,n_iters,processNoiseList[1],paramNoiseList[1],sim_save_name,1)
    try mkdir(data_folder)
    end
    cd(data_folder)
    open(string(sim_save,".txt"), "w") do f
        write(f,string("Sim save: ",sim_save,"\n"))
        write(f,string("CE settings: ",CEset,"\n"))
        write(f,string("PMAP input: ",pmapInput,"\n"))
    end
    open(string(sim_save,"_overall.txt"),"w") do g
        write(g,string("Init: ",CEset,"\n"))
    end
    cd("..")
else
    CE_iters = 1
    # combine the total name for saving
    for PRN in processNoiseList
        for PMN in paramNoiseList
            push!(noiseList,(PRN,PMN))
        end
    end
    pmapInput = noiseList
end
