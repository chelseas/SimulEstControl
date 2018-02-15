### Constructing the MDP

#from example: http://nbviewer.jupyter.org/github/sisl/POMDPs.jl/blob/master/examples/GridWorld.ipynb
# typealias EKFState MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

# MvNormal is not a concrete type --> might slow it down --> replace with FullNormal for faster
type MassMDP <: MDP{EKFState, Array{Float64,1}} # POMD{State,Action,Observation(1x2 hardcode)}
    discount_factor::Float64 # default 0.99
    goal_state::Float64
    force_range::LinSpace{Float64}
end

### Default constructor for MDP
function MassMDP()
    return MassMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
end

# initializing to default
mdp = MassMDP()

# redefining == for checking EKF states with logic
import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)
# redefining hash for using with EKF states
import Base.hash #might not need
hash(s::EKFState, h::UInt=zero(UInt)) = hash(mean(s), hash(cov(s), h))

### Finished constructing the MDP
create_state(::MassMDP) = MvNormal(zeros(ssm.nx),eye(ssm.nx))
create_observation(::MassMDP) = zeros(ssm.states)

function transition(mdp::MassMDP,s::EKFState,a::Array{Float64,1})
    obs = observation(mdp,s,a,s)
    if ukf_flag
      sp = ukf(ssm,obs,s,Q,R,a) # this is using UKF and causing divergence during simulation ERR
    else
      sp = filter(ssm, obs, s, Q, R, a)#was [a] before DIF for EKF
    end
    return sp
end

### Take a stochastic observation of the current state
function observation(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState)
    if state_mean
        x_assume = mean(s)
    else
        x_assume = rand(s)
    end
    x_p = ssm.f(x_assume,a) # WHY AM I PROPAGATING THIS?
    obs =  ssm.h(x_p,a)
    return obs
end

if prob == "2D"
  ### Calculate the reward for the current state and action
  function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState)
      if reward_type == "L1"
          r = sum(abs.(mean(s))'*-Qg) + sum(abs.(a)'*-Rg)
      elseif reward_type == "region"
          ms = mean(s)
          if (ms[4] > region_lb[1]) && (ms[5] > region_lb[2]) && (ms[6] > region_lb[3]) && (ms[4] < region_ub[1]) && (ms[5] < region_ub[2]) && (ms[6] < region_ub[3])
              r = rew_in_region
          else
              r = rew_out_region
          end
      end
      #@show r
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
end

### Checking for terminal case to stop solving --> setting so it never stops
# No reason to terminate early, want it to control continuously for given horizon
function POMDPs.isterminal(mdp::MassMDP, s::EKFState) # just for a leaf --> reach terminal state won't
  # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF
  if trace(cov(s)) > cov_thresh # add or for the estimates below thresh
      #@show "out"
      return true
  end
  #=
  if bounds
    # check if the next state given bounds are exceeded and terminate
    total_bounds = overall_bounds([-100.0],s,zeros(ssm.nu),w_bound) # can I check if the state after the action is within
    if desired_bounds < total_bounds
      return true
    end
  end
  =#
  return false
end

# Set discount factor to that defined in MDP abstract
POMDPs.discount(mdp::MassMDP) = mdp.discount_factor

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::MassMDP, s::EKFState, a::Array{Float64,1}, rng::AbstractRNG, sp::EKFState = create_state(mdp))
    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

if rollout == "random" && bounds == true # add this?
  type RandomController <: Policy # Policy{MvNormal}
      gain::Float64
  end
  function POMDPs.action(policy::RandomController, x::EKFState, a::Array{Float64,1}=zeros(ssm.nu))
      #xAssume = mean(x)
      #return policy.gain*xAssume[4:6]#,xAssume[5],xAssume[6]] #reason for using this variable?
      next_bound = desired_bounds + 1.0 # make more so it isn't true initially
      action_count = 0
      u_sample = zeros(ssm.nu)
      while (next_bound > desired_bounds) && (action_count < max_action_count)
        u_sample = fRange*(2*rand()-1)*ones(ssm.nu)
        next_bound = overall_bounds([-100.0],x,u_sample,w_bound) # can I check if the state after the action is within
        action_count = action_count + 1
      end
      if action_count == max_action_count
          global action_limit_count = action_limit_count + 1
      end
      return u_sample
  end
  roll = RandomController(pos_control_gain)
  heur = nothing
elseif rollout == "mpc" || rollout == "mpc2" # estimate value somehow to improve
    rollout_policy = FunctionPolicy() do s
        return MPCAction(s,depths)
    end

    type MyHeuristic # to be used to pass depth to MPC
        depth::Int64
    end
    function MCTS.next_action(h::MyHeuristic, mdp::MassMDP, s::EKFState, snode::DPWStateNode)
        return MPCAction(s,h.depth)
    end
    heur = MyHeuristic(depths)
end
