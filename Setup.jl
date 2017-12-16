
# SIM SETTINGS
prob = "2D" # set to the "1D" or "2D" problems defined
sim = "qmdp" # mcts, mpc, qmdp, drqn
rollout = "random" # MCTS/QMDP: random/position, DRQN: train/test
bounds = false # set bounds for mcts solver
quick_run = true
numtrials = 2 # number of simulation runs
noiseList = []
cond1 = "full"

#NOISE SETTINGS
processNoiseList = [0.0033]#[0.033, 0.1]#[0.001,0.0033,0.01,0.033,0.1,0.33] # default to full
paramNoiseList = [0.1,0.3]#,0.5,0.7]
ukf_flag = true # use ukf as the update method when computing mcts predictions
param_change = false # add a cosine term to the unknown param updates
param_type = "none" # sine or steps
param_magn = 0.2 # magnitude of cosine additive term # use >0.6 for steps
param_freq = 0.3

# Output settings
printing = false # set to true to print simple information
print_iters = true
plotting = false # set to true to output plots of the data
saving = false # set to true to save simulation data to a folder # MCTS trial at ~500 iters is 6 min ea, 1hr for 10
tree_vis = false # visual MCTS tree
sim_save = "CE2" # name appended to sim settings for simulation folder to store data from runs
data_folder = "bounds_test2"
fullobs = true # set to false for mpc without full obs
if sim != "mpc" # set fullobs false for any other sim
  fullobs = false
end

# CROSS ENTROPY SETTINGS
cross_entropy = true
save_last = false
num_pop = 6 #  number of samples to test this round of CE
num_elite = 6 # number of elite samples to keep to form next distribution
CE_iters = 3 # number of iterations for cross entropy
CE_params = 4 # number of params being sampled
states_m = 10.0
states_std = 5.0
act_m = 30.0
act_std = 20.0
depth_m = 15.0
depth_std = 10.0
expl_m = 50.0
expl_std = 25.0

# Settings for simulation
measNoise = 0.000001 # standard deviation of measurement noise
deltaT = 0.1 # timestep for simulation --> decrease for complex systems?
debug_bounds = false # set to 1 to print out commands to see when limits of state and estimation hit
cov_thresh = 1000 # threshold where trace(cov) of estimate will be discarded in MCTS
state_init = 1.0 # gain for teh initial state
state_min_tol = 0.1 # prevent states from growing less than X% of original value
friction_lim = 3.0 # limit to 2D friction case to prevent exploding growth

# settings for mcts
n_iters = 1000#3000#00 # total number of iterations
samples_per_state = 5#3 # want to be small
samples_per_act = 20 # want this to be ~20
depths = 20 # depth of tree
expl_constant = 10.0#100.0 #exploration const

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


if bounds
  include("EllipseBounds.jl")
  # will need to precompute this for each ProcessNoise case in the sim file so add it to main simulation.jl soon #TODO
  @show desired_bounds = 7.0#norm(1.2*ones(ssm.nx,1)) # setting for the limit to the ||Xt+1|| (maybe make in addition to the previous state?)
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
        temp_CE[1] = max(1.0,floor(temp_CE[1]))
        temp_CE[2] = max(1.0,floor(temp_CE[2]))
        temp_CE[3] = max(1.0,floor(temp_CE[3]))
        temp_CE[4] = max(0.1,floor(temp_CE[4]))

        push!(output,(temp_CE[1],temp_CE[2],temp_CE[3],temp_CE[4],iters,process,param,name,i,CE_count))
    end
    return output
end

if cross_entropy
    CEset_list = [states_m,states_std,act_m,act_std,depth_m,depth_std,expl_m,expl_std]
    CEset = MvNormal([states_m,act_m,depth_m,expl_m],diagm([states_std^2,act_std^2,depth_std^2,expl_std^2]))
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
