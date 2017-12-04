@auto_hash_equals struct LiteState
# discrete distrib over hidden param

# observedMeans::Array{Float64, 1}   # means of "observable" states
estimState::EKFState         # keep track of estimate over ALL states
hiddenVal::Float64    # theta, remains constant

end

# define constructors for LiteState

# --- Define LiteMDP --- #

type LiteMDP <: MDP{LiteState, Array{Float64, 1}}
  discount_factor::Float64 # default 0.99
  goal_state::Float64
  force_range::LinSpace{Float64}
  hiddenDiscrete::Array{Float64, 1}  # hidden params
  hiddenBelief::Array{Float64, 1}  # this is a belief over hiddenDiscrete
end

# define constructor and instantiate LiteMDP
function LiteMDP()

# state_init = 1
hiddenDiscrete = 0.1*state_init:0.1*state_init:1.9*state_init
################
  # 1D case:
  discreteStates = []
  discreteObs = []
  if prob == "1D"
    discreteStates =  [(x, y) for x in -2:0.2:2, y in -2:0.2:2]
    discreteObs = discreteStates
  end

  # initialize p(x'|x,th,a) storage as dictionary, gets updated in transitions
  prob_sp_th = Dict()
  for ds in discreteStates
    for th in hiddenDiscrete
      prob_sp_th[string((ds,th))] = 0
    end
  end
################

# check if paramNoise correct value to pass in
hiddenDistrib = Normal(state_init, paramNoise)

hiddenBelief = ones(length(hiddenDiscrete))

# what is the belief at this first value?
hiddenBelief[1] = cdf(hiddenDistrib, hiddenDiscrete[1])

# or set to uniform?
# need to debug this: a bit skewed
for i = 2:length(hiddenDiscrete)
    hiddenBelief[i] = cdf(hiddenDistrib, hiddenDiscrete[i]) - cdf(hiddenDistrib, hiddenDiscrete[i-1])
end

return LiteMDP(0.99,0.0,fDist, hiddenDiscrete, hiddenBelief)
end

mdp = LiteMDP()

#####################################

# --- Operator Overloading to Check Equality of EKF States --- #

# adding == method for checking EKF states with logic
import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)

# NOTE: == method is added for checking equality of LiteStates
#       using @auto_hash_equals macro up above in LiteState definition
#       DO NOT remove EKF == method up above


# ---- Create State and Observation Space ---- #

# default values fed into other functions

# NEED TO CHANGE THIS TO RETURN LiteSTATE
########
# how do we want to adapt this for different models? because there will be different numbers
  # of hidden vs observed states and ssm only gives one length for all states
# create_state(::LiteMDP) = LiteState(zeros(ssm.nx),MvNormal(zeros(ssm.nx),eye(ssm.nx)))
# create_observation(::LiteMDP) = zeros(ssm.states)  # this is not even used in this file?


# ---- Define Transition and Observation Distributions --- #

# change this to check whether we are belief state or not

# Vincent will do
# function transition(mdp::LiteMDP,s::LiteState,a::Array{Float64,1})
# end


# Vincent will do
#function observation(mdp::LiteMDP,s::LiteState,a::Array{Float64,1})
#end


# ---- Define Reward and Actions ---- #


### Calculate the reward for the current state and action
# Qg defined in Setup.jl file
function POMDPs.reward(mdp::LiteMDP,s::LiteState,a::Array{Float64,1},sp::LiteState)
  # reward + reward bonus
  # reward = expectation over hidden variable
  # reward from before: r = sum(abs.(s.trueState)'*-Qg) + sum(abs.(a)'*-Rg)
  # return r
  r = 0
  # loop over all possible values of theta
  for i = 1:length(mdp.hiddenDiscrete)
    theta = mdp.hiddenDiscrete[i]
    full_state = mean(s.estimState)
    obs_state = full_state[1:ssm.states]
    test_state = vcat(obs_state, theta)
    r_s = sum(abs.(test_state)'*-Qg) + sum(abs.(a)'*-Rg)
    r = r + mdp.hiddenBelief[i]*r_s
  end
  rb = 0
  for ds in mdp.discreteStates
    for i = 1:length(mdp.hiddenDiscrete)
      theta = mdp.hiddenDscrete[i]
      b = mdp.hiddenBelief[i]
      b_prime = mdp.hiddenBeliefNew[i]
      prob_sp = mdp.prob_sp_th[string((ds,theta))]
      rb = rb + prb_sp*b*abs(b_prime-b)
  lambda = 0.5
  return (r + lambda*rb)
end

if prob == "2D"
  # Defining action space for input controls
  immutable FFTActionSpace
      lower::Float64
      upper::Float64
  end

  ### Defines the action space as continuous in range of fRange
  function POMDPs.actions(mdp::LiteMDP)
      #take rand action within force bounds
      return FFTActionSpace(-fRange, fRange)#, mdp.force_range, mdp.force_range]
  end

  #Define POMDPs actions with function
  POMDPs.actions(mdp::LiteMDP, s::LiteState, as::Array{Float64,1}) = actions(mdp)

  ### Define rand for using within the range defined in FFT
  function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
      diff = as.upper-as.lower
      return diff*rand(rng, ssm.nu)+as.lower
  end

# -- 1D Case --- #
elseif prob == "1D"

  function POMDPs.actions(mdp::LiteMDP)
      #take rand action within force bounds
      return mdp.force_range
  end

  POMDPs.actions(mdp::LiteMDP, s::LiteState, as::Normal{Float64})  = actions(mdp)

  #to use default random rollout policy implement action sampling funct
  function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
     return rand(action_space)
  end
end


# ---- Check Terminal Case --- #

### Checking for terminal case to stop solving --> setting so it never stops
# No reason to terminate early, want it to control continuously for given horizon
function POMDPs.isterminal(mdp::LiteMDP, s::LiteState) # just for a leaf --> reach terminal state won't
  # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF

  # DEAL WITH BELIEF STATE/TRUE STATE CASES

  if isnull(s.beliefState)

    if s.trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
      return true
    end

  else# if trace(cov(s.beliefState)) > cov_thresh # add or for the estimates below thresh

    # crutch for now, cov cannot take in Nullable{EKFState}
    trueState = mean(get(s.beliefState))
    if trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
      return true
    end
  end

  return false

end


# --- Define Discount --- #

# Set discount factor to that defined in MDP abstract
POMDPs.discount(mdp::LiteMDP) = mdp.discount_factor  # not even used?


# --- Generate Next State and Reward ---- #

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::LiteMDP, s::LiteState, a::Array{Float64,1}, rng::AbstractRNG, sp::LiteState = create_state(mdp))

    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

# ---- Define Action --- # in our case, we just do random rollout for MCTS?

# don't need to define an action fxn; default action() in MCTS will sample
# random action from FFTActionSpace
