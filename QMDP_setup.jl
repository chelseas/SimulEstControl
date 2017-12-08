# ------------ SETUP FILE FOR QMDP ------------- #
# using StaticArrays
using AutoHashEquals

# NOTE: EKFState is still used for ukf
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

# --- Define Augmented State Type --- #
# allows us to check if using using belief state or not

@auto_hash_equals struct AugState  # struct = immutable, type = mutable
  beliefState::Nullable{EKFState}
  trueState::Array{Float64,1}
  # trueState::SVector{ssm.nx}  # from StaticArrays for better memory efficiency
end

# define constructors for convenience
AugState(bs::EKFState) = AugState(Nullable(bs), [])  # have empty trueState whenever not null
AugState(ts::Array{Float64,1}) = AugState(Nullable{EKFState}(), ts)


# --- Define Augmented MDP for QMDP --- #

if prob == "Car"
  type AugMDP <: MDP{AugState, Array{Float64, 1}}
    discount_factor::Float64
    goal_state::Float64
    force_range_1::LinSpace{Float64}
    force_range_2::LinSpace{Float64}
  end
elseif prob == "1D" || prob == "2D"
  type AugMDP <: MDP{AugState, Array{Float64, 1}}
    discount_factor::Float64 # default 0.99
    goal_state::Float64
    force_range::LinSpace{Float64}
  end
end

# --- Define Constructor and Instantiate AugMDP --- #
# constructor

if prob == "Car"
  function AugMDP()
    return AugMDP(0.99, 0.0, fDist[1], fDist[2])
  end
elseif prob == "1D" || prob == "2D"
  function AugMDP()
    return AugMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
  end
end

#show(fDist)
mdp = AugMDP()


# --- Operator Overloading to Check Equality of EKF States --- #

# adding == method for checking EKF states with logic
import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)

# NOTE: == method is added for checking equality of AugStates
#       using @auto_hash_equals macro up above in AugState definition
#       DO NOT remove EKF == method up above


# ---- Create State and Observation Space ---- #

# default values fed into other functions

# NEED TO CHANGE THIS TO RETURN AUGSTATE
create_state(::AugMDP) = AugState(zeros(ssm.nx))
# create_observation(::AugMDP) = zeros(ssm.states)  # this is not even used in this file?


# ---- Define Transition and Observation Distributions --- #

# change this to check whether we are belief state or not

# need to redefine for 2D case with a as a vector
function transition(mdp::AugMDP,s::AugState,a::Array{Float64,1})

  # IF "TRUE" STATE
  if isnull(s.beliefState)

    # regular MDP setup
    trueState = ssm.f(s.trueState, a)

    sp = AugState(trueState)  # return AugState with no belief

  # IF BELIEF STATE
  else

    # get(n, default): n is nullable, default if null
    x_assume = rand(get(s.beliefState))  # looks at whole distrib, not just mean
    x_p = ssm.f(x_assume,a)

    xPred = AugState(x_p)  # have to pass in AugState to observation
    obs = observation(mdp,xPred,a)  # redefine obs()? here are unused vars?

    if ukf_flag  # USE UKF

      belief = ukf(ssm,obs,get(s.beliefState),Q,R,a)  # assume 2D case, so [a] --> a
      trueState = mean(belief)
      sp = AugState(trueState)  # return AugState with no belief

    # USE EKF
    else

      belief = filter(ssm, obs, get(s.beliefState), Q, R, a)  # assume 2D case, so [a] --> a
      trueState = mean(belief)
      sp = AugState(trueState)

    end

  end

  return sp

end

# ask Zach if we can get rid of sp::AugState here
function observation(mdp::AugMDP,s::AugState,a::Array{Float64,1})

  # potentially use x_assume=rand(s.trueState) to get most out of uncertainty
  obs = ssm.h(s.trueState, a) # no measurement noise
  return obs
end


# ---- Define Reward and Actions ---- #

if prob == "2D"
  ### Calculate the reward for the current state and action
  function POMDPs.reward(mdp::AugMDP,s::AugState,a::Array{Float64,1},sp::AugState)

    if isnull(s.beliefState)

      r = sum(abs.(s.trueState)'*-Qg) + sum(abs.(a)'*-Rg)

    else

      trueState = mean(get(s.beliefState))
      r = sum(abs.(trueState)'*-Qg) + sum(abs.(a)'*-Rg)

    end

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
  POMDPs.actions(mdp::AugMDP, s::AugState, as::Array{Float64,1}) = actions(mdp)

  ### Define rand for using within the range defined in FFT
  function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
      diff = as.upper-as.lower
      return diff*rand(rng, ssm.nu)+as.lower
  end

elseif prob == "1D"

  function POMDPs.reward(mdp::AugMDP,s::AugState,a::Array{Float64,1}, sp::AugState)


    if isnull(s.beliefState)

      r = sum(abs.(s.trueState[1:ssm.states])'*-Qg) + sum(abs.(a)'*-Rg)

    else

      trueState = mean(get(s.beliefState))
      r = sum(abs.(trueState[1:ssm.states])'*-Qg) + sum(abs.(a)'*-Rg)

    end

    return r

  end

  function POMDPs.actions(mdp::AugMDP)
    #take rand action within force bounds
    return collect([f] for f in mdp.force_range)
  end

  POMDPs.actions(mdp::AugMDP, s::AugState, as::Normal{Float64})  = actions(mdp)

  #to use default random rollout policy implement action sampling funct
  function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
   return rand(action_space)
  end

elseif prob == "Car"

  function POMDPs.reward(mdp::AugMDP, s::AugState, a::Array{Float64, 1}, sp::AugState)

    # Get state estimate
    if isnull(s.beliefState)
      TrueState = s.trueState;
    else
      TrueState = mean(get(s.beliefState))
    end
    r = Car_reward(TrueState[:], a[:], TrackIdx[end])
    return r
  end

  # Define action space
  immutable FFTActionSpace
      lower::Array{Float64, 1}
      upper::Array{Float64, 1}
  end

  # Defines the action space as continuous in range of fRange
  function POMDPs.actions(mdp::AugMDP)
      # take rand action within force bounds
      return FFTActionSpace(-fRange, fRange) # frange is [u1 range; u2 range]
  end

  # Define POMDPs actions with function
  POMDPs.actions(mdp::AugMDP, s::AugState, as::Array{Float64,1}) = actions(mdp)

  ### Define rand for using within the range defined in FFT
  function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
      diff = as.upper-as.lower
      return diff.*rand(rng, ssm.nu) + as.lower
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
  else# if trace(cov(s.beliefState)) > cov_thresh # add or for the estimates below thresh
    # crutch for now, cov cannot take in Nullable{EKFState}
    trueState = mean(get(s.beliefState))
    if trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
      return true
    end
  end
  return false

#  if prob == "Car"
#    if isnull(s.beliefState)
#      Dist2TrackPoint = sqrt((s.trueState[1] - PathX[TrackIdx[end]])^2 + (s.trueState[2] - PathY[TrackIdx[end]])^2);
#      if (Dist2TrackPoint < dist_thresh)
#        return true
#      else
#        return false
#      end
#    else
#      Dist2TrackPoint = sqrt((mean(get(s.beliefState))[1] - PathX[TrackIdx[end]])^2 + (mean(get(s.beliefState))[2] - PathY[TrackIdx[end]])^2);
#        return true
#      else
#        return false
#      end
#    end
#  else
#    # DEAL WITH BELIEF STATE/TRUE STATE CASES
#    if isnull(s.beliefState)
#      if s.trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
#        return true
#      end
#    else# if trace(cov(s.beliefState)) > cov_thresh # add or for the estimates below thresh
#      # crutch for now, cov cannot take in Nullable{EKFState}
#      trueState = mean(get(s.beliefState))
#      if trueState[2] > 8000000000.0 #(mean(s)[2] < mdp.goal_state) || (mean(s)[2] > 80.0)) #position is past the goal state
#        return true
#      end
#    end
#    return false
#  end
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

# don't need to define an action fxn; default action() in MCTS will sample
# random action from FFTActionSpace
