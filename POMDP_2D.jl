### Constructing the MDP

#from example: http://nbviewer.jupyter.org/github/sisl/POMDPs.jl/blob/master/examples/GridWorld.ipynb
# typealias EKFState MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}


type MassMDP <: MDP{MvNormal, Array{Float64,1}} # POMD{State,Action,Observation(1x2 hardcode)}
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
    sp = filter(ssm, obs, s, Q, R, a)#was [a] before DIF for EKF
    #sp = ukf(ssm,obs,s,Q,R,a) # this is using UKF and causing divergence during simulation ERR
    return sp
end

### Take a stochastic observation of the current state
function observation(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState) # do i need to call rand here or does it do it for me?
    #sample the multivariate gaussian -- #"true val" for pos vel mass
    x_assume = rand(s) #rand(s) <-- should be sampled randomly using rand from MvNormal
    x_p = ssm.f(x_assume,a) #+ rand(mdp.w_noise)
    obs =  ssm.h(x_p,a) #+ rand(mdp.v_noise)# assuming sp updated prev, arg for ssm.h need a? did I pass in v correct?
    return obs
end

if prob == "2D"
  ### Calculate the reward for the current state and action
  function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState) #is s or sp the where i should pull x from?
      #gu = -60.0
      #Qg = -1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 70.0, 0.0, 0.0, 0.0, 0.0, 0.0])#-10.0, -10.0, -50.0])
      #R = -0.3*diagm([10.0, 10.0, 8.0]) #T was 30 before
      r = sum(abs.(mean(s))'*-Qg) + sum(abs.(a)'*-Rg)# + trace(cov(s))^2*gu#+ trace(diagm(diag(cov(s))[7:11]))^2*gu#trace(cov(s))^2*gu

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
  function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Array{Float64,1},sp::EKFState) #is s or sp the where i should pull x from?
      #can define these constants in MassPOMDP for more efficiency?
      #ga = -.008#0.01  -10 w/ fDist at 50; #gain for force cost  #force^2 maxes at 25000
      #gu = -10#-50;5 #gain for uncertainty #trace^2 maxes at 9
      #gp = -10#3   -20; #gain for position    #pos^2 starts 400 goes to 0
      #gv = -0.3#-0.3 normally, -10/-3 for good velocity tracking; #gain for veloc    #veloc^2 ranges 0 to 250s
      #ga = -1#0.01  -10 w/ fDist at 50; #gain for force cost  #force^2 maxes at 25000
      #gp = -10#3   -20; #gain for position    #pos^2 starts 400 goes to 0
      #gv = -3
      (gv, gp) = diag(-Qg)
      r = abs(mean(s)[2])*gp + abs(mean(s)[1])*gv + abs(a)*-Rg[1]# trace(cov(s))^2*gu +
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

# Set discount factor to that defined in MDP abstract
POMDPs.discount(mdp::MassMDP) = mdp.discount_factor

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::MassMDP, s::EKFState, a::Array{Float64,1}, rng::AbstractRNG, sp::EKFState = create_state(mdp))
    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

if rollout == "position"
  # Position controller to set the gain for roll-out
  type PositionController <: Policy # Policy{MvNormal}
      gain::Float64
  end
  ### action function --> using the average of the EKF state to select action
  function POMDPs.action(policy::PositionController, x::EKFState, a::Array{Float64,1}=zeros(ssm.nu))
      xAssume = mean(x) # I think deterministic is optimal
      return policy.gain*xAssume[pos_range]
  end
  roll = PositionController(pos_control_gain)
  heur = nothing
elseif rollout == "random"
  type RandomController <: Policy # Policy{MvNormal}
  end
  function POMDPs.action(policy::RandomController, x::EKFState, a::Array{Float64,1}=zeros(ssm.nu))
      #xAssume = mean(x)
      #return policy.gain*xAssume[4:6]#,xAssume[5],xAssume[6]] #reason for using this variable?
      return fRange*(2*rand()-1) # is this defined somewhere? fix this
  end
  roll = RandomController()
  heur = nothing
elseif rollout == "smooth"
  type SmoothController <: Policy # Policy{MvNormal}
      step::Float64
  end
  function POMDPs.action(policy::SmoothController, x::EKFState, a::Array{Float64,1}=zeros(ssm.nu))
      xAssume = mean(x)
      return policy.step*xAssume[pos_range]#,xAssume[5],xAssume[6]] #reason for using this variable?
  end
  roll = SmoothController(control_stepsize) # how can I control this?

  # This lets you get information from the simulation to pass into the next_action to choose actions
  mutable struct actHeur
    prev_action # previous action
    eta # bounds to stay within
  end
  # This function decides how to select the next action in the MCTS simulation is selected --> can bound
  # make wrapper for MDP called rate limiter actuator and add the actuator state to the MDP, where actuator state is just the prev action
  # then the action function can just pick random around it.This is cuz need prev state for each child in rollout sims. Not just from main loop
  function MCTS.next_action(h::actHeur, mdp::MassMDP, s::EKFState, snode::DPWStateNode)
      # bound the returned action to be random within the eta of h
      delta_a = (rand(3,1)*2-1)*h.eta # change to action is rand btw -1&1 * eta allowand
      return h.prev_action+delta_a
  end
  heur = actHeur(zeros(ssm.nu), control_stepsize)
end
