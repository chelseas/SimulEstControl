#from example: http://nbviewer.jupyter.org/github/sisl/POMDPs.jl/blob/master/examples/GridWorld.ipynb
# typealias EKFState MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}
const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}

type MassMDP <: MDP{MvNormal, Float64} # POMD{State,Action,Observation(1x2 hardcode)}
    discount_factor::Float64 # default 0.99
    goal_state::Float64
    force_range::LinSpace{Float64}
    #v_noise::MvNormal{Float64}
    #w_noise::MvNormal{Float64}
end

#default constructor
function MassMDP()
    return MassMDP(0.99,0.0,fDist)#,ssm.v,ssm.w)
end

mdp = MassMDP()

import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)

import Base.hash #might not need
hash(s::EKFState, h::UInt=zero(UInt)) = hash(mean(s), hash(cov(s), h))

#implementing the functions for GenerativeModels: https://github.com/JuliaPOMDP/GenerativeModels.jl/blob/master/src/GenerativeModels.jl
#good example: https://github.com/JuliaPOMDP/POMDPModels.jl/blob/master/src/InvertedPendulum.jl
create_state(::MassMDP) = MvNormal(zeros(ssm.nx),eye(ssm.nx))
#create_action(::MassMDP) = 0.0
create_observation(::MassMDP) = zeros(ssm.states)

function transition(mdp::MassMDP,s::EKFState,a::Float64)
    obs = observation(mdp,s,a,s)
    sp = filter(ssm, obs, s, Q, R,[a]) # will this get the correct w and v from global context?
    #sp = ukf(ssm,obs,s,Q,R, [a]) # is it correct to pass cov(w and v) here rather than s cov?
    return sp
end

function observation(mdp::MassMDP,s::EKFState,a::Float64,sp::EKFState) # do i need to call rand here or does it do it for me?
    #sample the multivariate gaussian -- #"true val" for pos vel mass
    x_assume = mean(s) #sample this randomly
    x_p = ssm.f(x_assume,a) #+ rand(mdp.w_noise)
    obs =  ssm.h(x_p,a) #+ rand(mdp.v_noise)# assuming sp updated prev, arg for ssm.h need a? did I pass in v correct?
    return obs
end

function POMDPs.reward(mdp::MassMDP,s::EKFState,a::Float64,sp::EKFState) #is s or sp the where i should pull x from?
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

function POMDPs.isterminal(mdp::MassMDP, s::EKFState)
    # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF
    if trace(cov(s)) > cov_thresh # add or for the estimates below thresh
        #@show "out"
        return true
    end
    return false
end

POMDPs.discount(mdp::MassMDP) = mdp.discount_factor

#to use default random rollout policy implement action sampling funct
function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
   return rand(action_space)
end

#call the function below to return sp (MvNormal), reward Float(64)
function POMDPs.generate_sr(mdp::MassMDP, s::EKFState, a::Float64, rng::AbstractRNG, sp::EKFState = create_state(mdp))
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
  function POMDPs.action(policy::PositionController, x::EKFState, a::Int64=0)
      xAssume = mean(x) # I think deterministic is optimal
      return policy.gain*xAssume[pos_range]
  end
  roll = PositionController(pos_control_gain)
  heur = nothing
elseif rollout == "random"
  type RandomController <: Policy # Policy{MvNormal}
  end
  function POMDPs.action(policy::RandomController, x::EKFState, a::Int64=0)
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
  function POMDPs.action(policy::SmoothController, x::EKFState, a::Int64=0)
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
#=
type PositionController <: Policy # Policy{MvNormal}
    gain::Float64
end
function POMDPs.action(policy::PositionController, x::EKFState, a::Int64=0)
    xAssume = mean(x)
    return policy.gain*xAssume[2]
end
=#
