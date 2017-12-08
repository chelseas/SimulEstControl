
if quick_run
  nSamples = 5 # quick amount of steps for debug_bounds
else
  nSamples = 50
end

# Settings for simulation
measNoise = 0.000001 # standard deviation of measurement noise
deltaT = 0.1 # timestep for simulation --> decrease for complex systems?
debug_bounds = false # set to 1 to print out commands to see when limits of state and estimation hit
cov_thresh = 1000 # threshold where trace(cov) of estimate will be discarded in MCTS
state_init = 1.0 # gain for teh initial state
state_min_tol = 0.1 # prevent states from growing less than X% of original value
friction_lim = 3.0 # limit to 2D friction case to prevent exploding growth

if prob == "2D"
  # Settings for simulation
  fRange = 5.0 # bounds on controls allowed within +- fRange
  #est_init = 11 # gain for the initial estimate
  state_init = 1.0 # gain for teh initial state
  fDist_disc = 1000 # discrete points in fDist force linspace
  # Reward shaping
  Qg = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # for use in POMDP reward funct
  Qr = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0]) # for computing reward below just for measured states
  Rg = 0.25*diagm([10.0, 10.0, 10.0])
  # Qg = 1.0*eye(11,11)#*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # for use in POMDP reward funct
  # Qr = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 70.0]) # for computing reward below just for measured states
  # Rg = 1.0*eye(3,3)#0.3*diagm([10.0, 10.0, 8.0])
  if (sim == "mcts") || (sim == "qmdp")
    # Parameters for the POMDP
    #500 = 7s, 2000 = 30s, 5000 = 60s
    n_iters = 200#3000#00 # total number of iterations
    samples_per_state = 5#3 # want to be small
    samples_per_act = 20 # want this to be ~20
    depths = 20 # depth of tree
    expl_constant = 10.0#100.0 #exploration const
    alpha_act = 1.0/10.0 # alpha for action
    alpha_st = 1.0/20.0 # alpha for state
    k_act = samples_per_act/(n_iters^alpha_act) # k for action
    k_st = samples_per_state/(n_iters^alpha_st) # k for state
    pos_control_gain = -80.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "mpc"
    n = 50 # horizon steps
  elseif sim == "drqn"
    load_dir = "" # Path to load the saved file
    epsilon = 0.5 # percentage to use EKF to update belief and draw sample to train from
    training_epochs = 5 # counter for training epochs
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
elseif sim == "drqn"
  using Distributions, TensorFlow
  if rollout == "train"
      using Convex, SCS, ECOS
  elseif rollout == "test"
  end
end
#using ForwardDiff # for EKF
if saving
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
  elseif sim == "drqn"
    include("DRQN.jl")
    if rollout == "train"
        include("MPC_2D.jl")
    elseif rollout == "test"
    end
  end
end


if bounds
  include("EllipseBounds.jl")
  # will need to precompute this for each ProcessNoise case in the sim file so add it to main simulation.jl soon #TODO
  @show desired_bounds = 5.0#norm(1.2*ones(ssm.nx,1)) # setting for the limit to the ||Xt+1|| (maybe make in addition to the previous state?)
  n_w = 30
  n_out_w = 10
  F = 2.34
  w_bound_samples = ellipsoid_bounds(MvNormal(zeros(ssm.nx),processNoiseList[1]eye(ssm.nx,ssm.nx)),n_w,n_out_w,F) # precompute the w_bound
  w_bound_avg = mean(w_bound_samples,2) # average samples of the w_bound
  @show w_bound = norm(w_bound_avg*sqrt(processNoiseList[1])) # compute the norm of the STD * the w_bound
  max_action_count = 20
  action_count = 1
end

if tree_vis
  hist = HistoryRecorder() # necessary for POMDP setup #zach: is there a way to extract all actions and rewards for each step?
end
# solver defined here with all settings
if (sim == "mcts") || (sim == "qmdp")
  if rollout == "smooth"
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll), next_action=heur)#-4 before
  elseif rollout == "random" && bounds == true
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll))#, enable_tree_vis = tree_vis)#-4 before
  elseif rollout == "random"
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)#, enable_tree_vis = tree_vis)#-4 before
  else
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll))#-4 before
  end
  policy = solve(solver,mdp) # policy setup for POMDP
end
