
if run == "quick"
  nSamples = 70 # quick amount of steps for debug_bounds
elseif run == "long"
  nSamples = 220
else
  nSamples = 50
end

# Settings for simulation
measNoise = 0.000001 # standard deviation of measurement noise
deltaT = 0.1 # timestep for simulation --> decrease for complex systems?
debug_bounds = false # set to 1 to print out commands to see when limits of state and estimation hit
cov_thresh = 1000 # threshold where trace(cov) of estimate will be discarded in MCTS
state_init = 1.0 # gain for the initial state
state_min_tol = 0.1 # prevent states from growing less than X% of original value

if prob == "1D"
  # Settings for simulation
  fRange = 1.0 # bounds on controls allowed within +- fRange
  #est_init = 11 # gain for the initial estimate
  fDist_disc = 1000 # discrete points in fDist force linspace
  # Reward shaping
  Qg = diagm([3;20]) # diagm([3;10])  # [velocity, position]
  Qr = Qg
  Rg = [1]
  if (sim == "mcts") || (sim == "qmdp")
    # Parameters for the POMDP
    n_iters = 50 # total number of iterations
    depths = 5 # depth of tree
    expl_constant = 100.0 #exploration const
    k_act = 8.0 # k for action
    alpha_act = 1.0/5.0 # alpha for action
    k_st = 8.0 # k for state
    alpha_st = 1.0/5.0 # alpha for state
    pos_control_gain = -4.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "lite"
    n_iters = 50 # total number of iterations
    depths = 5 # depth of tree
    expl_constant = 100.0 #exploration const
    k_act = 8.0 # k for action
    alpha_act = 1.0/5.0 # alpha for action
    k_st = 8.0 # k for state
    alpha_st = 1.0/5.0 # alpha for state
    pos_control_gain = -4.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "mpc"
    n = 20 # horizon steps # using receding horizon now

  end
elseif prob == "2D"
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
    n_iters = 500 # total number of iterations
    depths = 20 # depth of tree
    expl_constant = 100.0 #exploration const
    k_act = 8.0 # k for action
    alpha_act = 1.0/5.0 # alpha for action
    k_st = 8.0 # k for state
    alpha_st = 1.0/5.0 # alpha for state
    pos_control_gain = -8.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "mpc"
    n = 50 # horizon steps
  end

elseif prob == "Car"
  fRange = [0.5; 20];
  fDist_disc = [1000; 1000];
  #Qg = 1.0*diagm([1.0, 1.0, 1.0, 1.0, 0.0])
  #Qr = 1.0*diagm([1.0, 1.0, 1.0, 1.0)]
  #Rg = 1.0*diagm([1.0, 1.0])
  Ktrack = 10.0
  Kdist = 1.0
  Kspeed = 1.0
  Kang = 1.0
  # Define shape of road -- in future, may want to add lane markers
  PathX = collect(0:0.1:100);
  LeftX = collect(0:0.1:101.5);
  RightX = collect(0:0.1:98.5);
  PathY = sqrt.(100^2 - PathX.^2);
  LeftLaneY = sqrt.(101.5^2 - LeftX.^2);
  RightLaneY = sqrt.(98.5^2 - RightX.^2);
  # Define speed limit -- speed matching goal
  SpeedLimit = 50.0
  x0 = PathX[1];
  y0 = PathY[1];
  theta0 = atan((PathY[2] - PathY[1])/(PathX[2] - PathX[1]));
  v0 = 0.0;
  mu0 = 0.1;

  TrackIdx = [1];
  if sim == "lite"
    point_lead = 1.0 # 10.0;
    dist_thresh = 0.5# 5.0;
  elseif sim == "qmdp" || sim == "mcts"
    point_lead = 10.0
    dist_thresh = 5.0
  end
  lead_dist = abs.(sqrt.((x0 - PathX[TrackIdx[end] + 1 : end]).^2 + (y0 - PathY[TrackIdx[end] + 1 : end]).^2) - point_lead);
  newDist, newIdx = findmin(lead_dist);
  append!(TrackIdx, newIdx);

  state_init = [x0; y0; theta0; v0; mu0];
  if (sim == "mcts") || (sim == "qmdp")
    # Parameters for the POMDP
    n_iters = 50 # total number of iterations
    depths = 5 # 20 # depth of tree
    expl_constant = 100.0 #exploration const
    k_act = 8.0 # k for action
    alpha_act = 1.0/5.0 # alpha for action
    k_st = 8.0 # k for state
    alpha_st = 1.0/5.0 # alpha for state
    pos_control_gain = -8.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "lite"
    n_iters = 50 # total number of iterations
    depths = 5 # depth of tree
    expl_constant = 100.0 #exploration const
    k_act = 8.0 # k for action
    alpha_act = 1.0/5.0 # alpha for action
    k_st = 8.0 # k for state
    alpha_st = 1.0/5.0 # alpha for state
    pos_control_gain = -8.0 # gain to drive position rollout --> higher = more aggressive
    control_stepsize = 5.0 # maximum change in control effort from previous action
  elseif sim == "mpc"
    n = 50 # horizon steps
  end
end

# Packages
if (sim == "mcts") || (sim == "qmdp") || (sim == "lite")
  using Distributions, POMDPs, MCTS, POMDPToolbox # for MCTS
elseif sim == "mpc"
  using Distributions, Convex, Mosek, SCS, ECOS# for MPC
  if fullobs
    rollout = "fobs"
  else
    rollout = "unk"
  end
end

using ForwardDiff # for EKF

if saving
  include("Save.jl")
end

if plotting
  # using PyPlot
  using Plots  # metapackage bringing lots of plotting packages together
  plotly()  # turns on plotly as backend
end
#using PyPlot #broken?

# First have to load SSM to define params for rest of the setup
include("SSM.jl") # contains SSM definitions and functions
if prob == "2D"
  ssm = build2DSSM(deltaT)#,processNoise,measNoise) # building state-space
  prob_params = ["vx","vy","w","x","y","theta","m","uv","J","rx","ry"]
  control_params = ["Fx", "Fy", "T"]
elseif prob == "Car"
  ssm = buildCarSSM(deltaT);
  prob_params = ["x", "y", "theta", "v", "mu"];
  control_params = ["u1", "u2"];
elseif prob == "1D"
  ssm = buildDoubleIntSSM(deltaT)#,processNoise,measNoise) # building 1D state-space
  prob_params = ["v","p","m"]
  control_params = ["u"]
end

# Force limits in the problem --> need before defining MPC/POMDP
if prob == "Car"
  Fvar = fRange;
  TVar = fRange;
  fDist = linspace(-fRange[1], fRange[1], fDist_disc[1]), linspace(-fRange[2], fRange[2], fDist_disc[2]);
  uDist = MvNormal(zeros(ssm.nu), diagm(fRange));
  #pos_range = 1:4;

  startState = 1;

else
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
end


# initialize first process and param noises and Q,R
processNoise = processNoiseList[1]
paramNoise = paramNoiseList[1]
Q = diagm(processNoise*ones(ssm.nx))
R = diagm(measNoise*ones(ssm.ny))

# Loading scripts based on simulation and problem conditions
if ukf_flag
  include("UKF.jl") # contains UKF
else
  include("EKF.jl") # contains EKF
end

if prob == "2D" # load files for 2D problem
  include("LimitChecks_2D.jl") # checks for control bounds and state/est values
  if sim == "mcts"
    include("POMDP_2D.jl") # functions for POMDP definition
  elseif sim == "qmdp"
    include("QMDP_setup.jl")
  elseif sim == "mpc"
    include("MPC_2D.jl") # function to set up MPC opt and solve
  end
elseif prob == "Car"
  include("LimitChecks_Car.jl")
  if sim == "qmdp"
    include("QMDP_setup.jl")
  elseif sim == "mcts"
    include("POMDP_2D.jl")
  elseif sim == "lite"
    include("POMDP_lite.jl")
  end

elseif prob == "1D" # load files for 1D problem
  include("LimitChecks_1D.jl") # checks for control bounds and state/est values
  if sim == "mcts"
    include("POMDP_1D.jl") # functions for POMDP definition
  elseif sim == "qmdp"
    include("QMDP_setup.jl")
  elseif sim == "lite"
    include("POMDP_lite.jl")
  elseif sim == "mpc"
    include("MPC_1D.jl") # function to set up MPC opt and solve
  end
end

# Initializing variables for the simulation to be stored
#=
obs = Array{Float64,2}(ssm.ny,nSamples) #measurement history
u = Array{Float64,2}(ssm.nu,nSamples) #input history
x = Array{Float64,2}(ssm.nx,nSamples+1) #state history
est = Array{Float64,2}(ssm.nx,nSamples+1) #store mean of state estimate
uncertainty = Array{Array{Float64,2},1}(nSamples+1) #store covariance of state estimate
rewrun = Array{Float64,1}(nSamples) # total reward summed for each run
=#

#hist = HistoryRecorder() # necessary for POMDP setup #zach: is there a way to extract all actions and rewards for each step?
# solver defined here with all settings
if (sim == "mcts") || (sim == "qmdp") || (sim == "lite")
  if rollout == "smooth"
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll), next_action=heur)#-4 before
  elseif rollout == "random"
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)#-4 before
  else
    solver = DPWSolver(n_iterations = n_iters, depth = depths, exploration_constant = expl_constant,
    k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st, estimate_value=RolloutEstimator(roll))#-4 before
  end
  policy = solve(solver,mdp) # policy setup for POMDP
end
