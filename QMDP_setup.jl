# ------------ SETUP FILE FOR QMDP ------------- #
using StaticArrays

# NOTE: EKFState is still used for ukf
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

# --- Define Augmented State Type --- #
# allows us to check if using using belief state or not
type AugState
  beliefState::Nullable{EKFState}()
  trueState::SVector  # from StaticArrays.jl
end

# --- Define Augmented MDP for QMDP --- #

type AugMDP <: MDP{AugState, Array{Float64, 1}}
  discount_factor::Float64 # default 0.99
  goal_state::Float64
  force_range::LinSpace{Float64}
end


# --- Define Constructor and Anstantiate AugMDP --- #
# constructor
function AugMDP()
  return AugMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
end

mdp = AugMDP()


# --- Operator Overloading to Check Equality of EKF States --- #

# redefining == for checking EKF states with logic
import Base.==2
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)
# redefining hash for using with EKF states
import Base.hash #might not need
hash(s::EKFState, h::UInt=zero(UInt)) = hash(mean(s), hash(cov(s), h))


# ---- Create State and Observation Space ---- # 

# default values fed into other functions
create_state(::AugMDP) = MvNormal(zeros(ssm.nx),eye(ssm.nx))
create_observation(::AugMDP) = zeros(ssm.states)


# ---- Define Transition and Observation Distributions --- #


# TO DO:
# - define a new state, depending on if current state is belief or not

# change this to check whether we are belief state or not
function transition(mdp::AugMDP,s::AugState,a::Float64)

  # check if "true state"
  if isnull(s.beliefState)

    # regular MDP setup
     sp = ssm.f(s.trueState, a)

  # check if belief state
  elseif 

    obs = observation(mdp,s,a,s)
    if ukf_flag
      sp = ukf(ssm,obs,s,Q,R,[a]) # this is using UKF and causing divergence during simulation ERR
    else
      sp = filter(ssm, obs, s, Q, R, [a]) #was [a] before DIF for EKF
    end
    return sp

  end

end

function observation(mdp::MassMDP,s::EKFState,a::Float64,sp::EKFState) # do i need to call rand here or does it do it for me?
    x_assume = rand(s)
    x_p = ssm.f(x_assume,a) # do I need noise here?
    obs =  ssm.h(x_p,a)
    return obs
end


# ---- Define Rewards and Actions ---- #
if prob == "2D"
  ### Calculate the reward for the current state and action
  function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState)
      r = sum(abs.(mean(s))'*-Qg) + sum(abs.(a)'*-Rg)
      return r
  end

  # Defining action space for input controls
  immutable FFTActionSpace
      lower::Float64
      upper::Float64
  end

  ### Defines the action space as continuous in range of fRange
  function POMDPs.actions(mdp::MassMDP)
      #take rand action within force bounds
      return FFTActionSpace(-fRange, fRange)#, mdp.force_range, mdp.force_range]
  end

  #Define POMDPs actions with function
  POMDPs.actions(mdp::MassMDP, s::EKFState, as::Array{Float64,1})  = actions(mdp)

  ### Define rand for using within the range defined in FFT
  function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
      diff = as.upper-as.lower
      return diff*rand(rng, ssm.nu)+as.lower
  end
elseif prob == "1D"
  function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState)
      (gv, gp) = diag(-Qg)
      r = abs(mean(s)[2])*gp + abs(mean(s)[1])*gv + abs(a)*-Rg[1]
      return r
  end
  function POMDPs.actions(mdp::MassMDP)
      #take rand action within force bounds
      return mdp.force_range
  end

  POMDPs.actions(mdp::MassMDP, s::EKFState, as::Normal{Float64})  = actions(mdp)
  #to use default random rollout policy implement action sampling funct
  function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
     return rand(action_space)
  end
end


# ---- Check Terminal Case --- #

### Checking for terminal case to stop solving --> setting so it never stops
# No reason to terminate early, want it to control continuously for given horizon
function POMDPs.isterminal(mdp::MassMDP, s::EKFState) # just for a leaf --> reach terminal state won't
  # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF
  if trace(cov(s)) > cov_thresh # add or for the estimates below thresh
      #@show "out"
      return true
  end
  return false
end


# --- Define Discount --- #

# Set discount factor to that defined in MDP abstract
POMDPs.discount(mdp::MassMDP) = mdp.discount_factor


# --- Generate Next State and Reward ---- #

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::MassMDP, s::EKFState, a::Array{Float64,1}, rng::AbstractRNG, sp::EKFState = create_state(mdp))
    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

# ---- Define Action --- # in our case, we just do random rollout for MCTS?



