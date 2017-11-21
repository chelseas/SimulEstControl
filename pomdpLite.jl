# ------------ SETUP FILE FOR QMDP ------------- #
# using StaticArrays
using AutoHashEquals

# NOTE: EKFState is still used for ukf
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

# --- Define Augmented State Type --- #
# allows us to check if using using belief state or not

@auto_hash_equals struct LiteState  # struct = immutable, type = mutable
  # hidState:: TODO -- distribution over discrete state Space
  obsState::Array{Float64,1}
  fullState::EKFState
  # trueState::SVector{ssm.nx}  # from StaticArrays for better memory efficiency
end

# define constructors for convenience
#LiteState(bs::EKFState) = LiteState(Nullable(bs), [])  # have empty trueState whenever not null
#LiteState(ts::Array{Float64,1}) = LiteState(Nullable{EKFState}(), ts)


# --- Define Augmented MDP for QMDP --- #

type LiteMDP <: MDP{LiteState, Array{Float64, 1}}
  discount_factor::Float64 # default 0.99
  goal_state::Float64
  force_range::LinSpace{Float64}
  # need to cache rewards for x,a pairs for faster computation here?
end


# --- Define Constructor and Instantiate LiteMDP --- #
# constructor
function LiteMDP()
  return LiteMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
end

mdp = LiteMDP()


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

if prob == "2D"
  ### Calculate the reward for the current state and action
  # Qg defined in Setup.jl file
  function POMDPs.reward(mdp::LiteMDP,s::LiteState,a::Array{Float64,1},sp::LiteState)
    # reward + reward bonus
    # reward = expectation over hidden variable
    # reward from before: r = sum(abs.(s.trueState)'*-Qg) + sum(abs.(a)'*-Rg)
    # return r
  end

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

  function POMDPs.reward(mdp::LiteMDP,s::LiteState,a::Float64,sp::LiteState)

      # DEAL WITH BELIEF AND TRUE STATE CASES

      (gv, gp) = diag(-Qg)
      r = abs(s.trueState[2])*gp + abs(s.trueState[1])*gv + abs(a)*-Rg[1]
      return r
  end

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
