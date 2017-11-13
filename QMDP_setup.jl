# TO DO
# this file:
#   - go every function, and make sure modified correctly
#   - EKFState --> AugState, ETC.
#   - define action fxn?
# simulation.jl
#   - do we need to use AugState for xNew? I think we do


# ------------ SETUP FILE FOR QMDP ------------- #
using StaticArrays

# NOTE: EKFState is still used for ukf
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

# --- Define Augmented State Type --- #
# allows us to check if using using belief state or not
type AugState
  beliefState::Nullable{EKFState}
  trueState::SVector{ssm.nx}  # from StaticArrays for better memory efficiency
end

# --- Define Augmented MDP for QMDP --- #

type AugMDP <: MDP{AugState, Array{Float64, 1}}
  discount_factor::Float64 # default 0.99
  goal_state::Float64
  force_range::LinSpace{Float64}
end


# --- Define Constructor and Instantiate AugMDP --- #
# constructor
function AugMDP()
  return AugMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
end

mdp = AugMDP()


# --- Operator Overloading to Check Equality of EKF States --- #

# redefining == for checking EKF states with logic
import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)
# redefining hash for using with EKF states
#import Base.hash #might not need
# hash(s::EKFState, h::UInt=zero(UInt)) = hash(mean(s), hash(cov(s), h))

# WILL THIS WORK?
import Base.==
==(a::AugState, b::AugState) = a.beliefState == b.beliefState && a.trueState == b.trueState


# ---- Create State and Observation Space ---- # 

# default values fed into other functions

# NEED TO CHANGE THIS TO RETURN AUGSTATE
create_state(::AugMDP) = AugState(Nullable{EKFState}(), zeros(ssm.nx))
create_observation(::AugMDP) = zeros(ssm.states)  # this is not even used in this file?


# ---- Define Transition and Observation Distributions --- #

# change this to check whether we are belief state or not
function transition(mdp::AugMDP,s::AugState,a::Float64)

  # IF "TRUE" STATE
  if isnull(s.beliefState)

    # regular MDP setup
    trueState = ssm.f(s.trueState, a)
    sp = AugState(Nullable{EKFState}(), trueState)  # return AugState with no belief

  # IF BELIEF STATE
  else 

    obs = observation(mdp,s,a,s)

    if ukf_flag  # USE UKF

      belief = ukf(ssm,obs,s.beliefState,Q,R,[a])
      trueState = mean(belief)
      sp = Augstate(Nullable{EKFState}(), trueState)  # return AugState with no belief

    else  # USE EKF

      belief = filter(ssm, obs, s.beliefState, Q, R, [a])
      trueState = mean(belief)
      sp = Augstate(Nullable{EKFState}(), trueState)

    end

  end

  return sp

end

function observation(mdp::AugMDP,s::AugState,a::Float64,sp::AugState)
  # x_assume = rand(s)
  # x_p = ssm.f(x_assume,a)
  # obs =  ssm.h(x_p,a)

  obs = ssm.h(s.trueState, a) # + noise? what is cosine/sine term in h()?
  return obs
end


# ---- Define Reward and Actions ---- #

if prob == "2D"
  ### Calculate the reward for the current state and action
  function POMDPs.reward(mdp::AugMDP,s::AugState,a::Array{Float64,1},sp::AugState)

      r = sum(abs.(s.trueState)'*-Qg) + sum(abs.(a)'*-Rg)
      return r
  end

  # Defining action space for input controls
  immutable FFTActionSpace
      lower::Float64
      upper::Float64
  end

  ### Defines the action space as continuous in range of fRange
  function POMDPs.actions(mdp::AugMDP)
      #take rand action within force bounds
      return FFTActionSpace(-fRange, fRange)#, mdp.force_range, mdp.force_range]
  end

  #Define POMDPs actions with function
  POMDPs.actions(mdp::AugMDP, s::AugState, as::Array{Float64,1})  = actions(mdp)

  ### Define rand for using within the range defined in FFT
  function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
      diff = as.upper-as.lower
      return diff*rand(rng, ssm.nu)+as.lower
  end

# -- 1D Case --- #
elseif prob == "1D"

  function POMDPs.reward(mdp::AugMDP,s::AugState,a::Array{Float64,1},sp::AugState)

      # DEAL WITH BELIEF AND TRUE STATE CASES

      (gv, gp) = diag(-Qg)
      r = abs(s.trueState[2])*gp + abs(s.trueState[1])*gv + abs(a)*-Rg[1]
      return r
  end

  function POMDPs.actions(mdp::AugMDP)
      #take rand action within force bounds
      return mdp.force_range
  end

  POMDPs.actions(mdp::AugMDP, s::AugState, as::Normal{Float64})  = actions(mdp)

  #to use default random rollout policy implement action sampling funct
  function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
     return rand(action_space)
  end
end


# ---- Check Terminal Case --- #

### Checking for terminal case to stop solving --> setting so it never stops
# No reason to terminate early, want it to control continuously for given horizon
function POMDPs.isterminal(mdp::AugMDP, s::AugState) # just for a leaf --> reach terminal state won't
  # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF

  # DEAL WITH BELIEF STATE/TRUE STATE CASES

  if isnull(s.beliefState)

    if s.trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
      return true
    end

  else #if trace(cov(s.beliefState)) > cov_thresh # add or for the estimates below thresh
    
    if s.trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
      return true
    end
  end

  return false

end


# --- Define Discount --- #

# Set discount factor to that defined in MDP abstract
POMDPs.discount(mdp::AugMDP) = mdp.discount_factor  # not even used?


# --- Generate Next State and Reward ---- #

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::AugMDP, s::AugState, a::Array{Float64,1}, rng::AbstractRNG, sp::AugState = create_state(mdp))

    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

# ---- Define Action --- # in our case, we just do random rollout for MCTS?

# do we need to define an action for random rollout? Patrick says no,
# already implemented in MCTS package, but I am skeptical

type RandomController <: Policy # Policy{MvNormal}
    gain::Float64
end

function POMDPs.action(policy::RandomController, x::AugState, a::Array{Float64,1}=zeros(ssm.nu))
    #xAssume = mean(x)
    #return policy.gain*xAssume[4:6]#,xAssume[5],xAssume[6]] #reason for using this variable?
    return rand()#fRange*(2*rand()-1) # is this defined somewhere? fix this
end

