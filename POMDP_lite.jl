# ---------- SETUP FILE FOR POMDP-lite ---------- #

using AutoHashEquals
using StatsBase

const EKFState = MvNormal{Float64,PDMats.PDMat{Float64,Array{Float64,2}},Array{Float64,1}}


# --- Define LiteState --- #

# let's assume 1 hidden parameter, friction coefficient
@auto_hash_equals struct LiteState

	estimState::EKFState  		       # keep track of estimate over ALL states
	hiddenVal::Float64 				   # theta, remains constant

end

# define constructors for LiteState

# --- Define LiteMDP --- #
if prob == "Car"
    type LiteMDP <: MDP{LiteState, Array{Float64, 1}}
        discount_factor::Float64
        goal_state::Float64
        force_range_1::LinSpace{Float64}
        force_range_2::LinSpace{Float64}
        hiddenDiscrete::Array{Float64, 1}  # hidden params
        discreteStates::Array{Array{Float64, 1}, 4}
        hiddenBelief::Array{Float64, 1}  # this is a belief over hiddenDiscrete
        hiddenBeliefNew::Array{Float64, 1}
        prob_sp_th::Dict
    end
else
    type LiteMDP <: MDP{LiteState, Array{Float64, 1}}
      discount_factor::Float64 # default 0.99
      goal_state::Float64
      force_range::LinSpace{Float64}
      hiddenDiscrete::Array{Float64, 1}  # hidden params
      discreteStates::Array{Array{Float64, 1}, 2}
      hiddenBelief::Array{Float64, 1}  # this is a belief over hiddenDiscrete
      hiddenBeliefNew::Array{Float64, 1}
      prob_sp_th::Dict
    end
end

# define constructor and instantiate LiteMDP
function LiteMDP()

	# state_init = 1
    if prob == "Car"
        hiddenDiscrete = 0.1*state_init[end]:0.1*state_init[end]:1.9*state_init[end]
        # check if paramNoise correct value to pass in
        hiddenDistrib = Normal(state_init[end], paramNoise)
    else
    	hiddenDiscrete = 0.1*state_init:0.1*state_init:1.9*state_init	# check if paramNoise correct value to pass in
    	hiddenDistrib = Normal(state_init, paramNoise)
    end

	hiddenBelief = ones(length(hiddenDiscrete))

	# what is the belief at this first value?
	hiddenBelief[1] = cdf(hiddenDistrib, hiddenDiscrete[1])

	for i = 2:length(hiddenDiscrete)
    	hiddenBelief[i] = cdf(hiddenDistrib, hiddenDiscrete[i]) - cdf(hiddenDistrib, hiddenDiscrete[i-1])
	end

	 # 1D case:
	discreteStates = []
	discreteObs = []
	sr = 2
	if prob == "1D"
		# discreteStates is a 2D array,
		# where every point is a tuple corresponding to a possible state
		discreteStates =  [[x, y] for x in -sr:1:sr, y in -sr:1:sr]
		# discreteStates = reshape(discreteStates, length(discreteStates))
		discreteObs = discreteStates
    elseif prob == "Car"
        # 4 observable States, posx, posy, th, v
        target_x = PathX[TrackIdx[end]]
        target_y = PathY[TrackIdx[end]]
        # println((target_x, target_y))
        #discreteStates = [[x1,x2,x3,x4] for x1 in target_x-11:2:target_x+1, x2 in target_y-1:1:target_y+11, x3 in -pi:pi/2:pi, x4 in -12:4:12]
        #discreteStates = [[x1,x2,x3,x4] for x1 in target_x-20:10:target_x+20, x2 in target_y-20:10:target_y+20, x3 in -pi:pi/2:pi, x4 in -12:4:12]
        discreteStates = [[x1,x2,x3,x4] for x1 in target_x-2:1:target_x+2, x2 in target_y-2:1:target_y+2, x3 in -pi:pi/2:pi, x4 in -12:4:12]
        # discreteStates = reshape(discreteStates,length(discreteStates))
        # println("discrete states ",discreteStates)
        discreteObs = discreteStates
	end

	# initialize p(x'|x,th,a) storage as dictionary, gets updated in transitions

	prob_sp_th = Dict()
	for ds in discreteStates
		for th in hiddenDiscrete
		  prob_sp_th[string((ds,th))] = 0.0
		end
	end

	# just putting hiddenBelief as placeholder for hiddenBeliefNew
    if prob == "Car"
        return LiteMDP(0.99,0.0,fDist[1],fDist[2],hiddenDiscrete,discreteStates,hiddenBelief, hiddenBelief,prob_sp_th)
    else
    	return LiteMDP(0.99,0.0,fDist, hiddenDiscrete, discreteStates, hiddenBelief, hiddenBelief, prob_sp_th)
    end
end

mdp = LiteMDP()


# --- Operator Overloading to Check Equality of LiteStates --- #

# adding == method for checking EKF states with logic
import Base.==
==(a::EKFState,b::EKFState) = mean(a) == mean(b) && cov(a) == cov(b)

# use @auto_hash_equals to check equality of LiteStates


# ---- Constructor for Creating LiteState ---- #

# not defining constructor (not a necessary POMDP function)

# --- Define Transitions & Observations --- #

function transition(mdp::LiteMDP,s::LiteState,a::Array{Float64,1})

	# For every xp in the discrete state matrix approximation,
	# x is the current mean of the ekf state (not including theta)
	# Sum over theta, plug theta into state,’
	# True xNew = propagated dynamics
	# P(xp|x,theta,a) = pdf(mvnormal is (mean xNew (not including theta), cov block diag section of ekf state), query is xp)

	if (prob == "1D") || (prob == "Car")
		probWeights = Array{Float64, 1}(length(mdp.discreteStates))
 		newStateDistribSum = 0
 		weightSum = 0

		x = mean(s.estimState)[1:ssm.states]
		xCov = 100*cov(s.estimState)[1:ssm.states, 1:ssm.states]  # take relevant block diagonal
        # println("xCov ",xCov)

 		# --- Determine Transition Probabilities to All Possible New States --- #
        xNew = x
		for ii = 1:length(mdp.discreteStates)
			xp = mdp.discreteStates[ii]
			prob_accum = 0
			for ind = 1:length(mdp.hiddenDiscrete)
				theta = mdp.hiddenDiscrete[ind]
				xTrue = vcat(x, theta)  # create "true state"
				xNew = ssm.f(xTrue, a)  # "propagate dynamics"
                # println("xNew ", xNew)
				xDistrib = MvNormal(xNew[1:ssm.states], xCov)  # create MV gaussian
				prob = pdf(xDistrib, xp)  # ASSUMPTION: approximate prob mass as prob density
                #println("prob ", prob)
				mdp.prob_sp_th[string((xp, theta))] = prob  #
				newStateDistribSum += prob
				prob_accum += mdp.hiddenBelief[ind]*prob
			end

			# append prob to probWeights
			probWeights[ii] = prob_accum
            # println("prob_accum ", prob_accum)
			weightSum += prob_accum
		end
		# --- Normalize Transition Probabilities --- #
		for key in keys(mdp.prob_sp_th)
			mdp.prob_sp_th[key] /= newStateDistribSum
		end

        xSample = x
        if weightSum < 0.000001
            # println("problem with weights")
            xSample = xNew[1:ssm.states]
        else
    		for jj = 1:length(probWeights)
    			probWeights[jj] /= weightSum
    		end

    		# --- Calculate Next State --- #
            # println("weights ", probWeights)
    		xSample = sample(mdp.discreteStates, Weights(probWeights))
        end
        #println("x ", x)
        #println("target ", (PathX[TrackIdx[end]],PathY[TrackIdx[end]]))
        #println("xSample, ", xSample)

		stateDistrib = MvNormal(vcat(xSample, s.hiddenVal), cov(s.estimState))

		obs = observation(mdp, s, a)

		if ukf_flag  # USE UKF
			belief = ukf(ssm, obs, stateDistrib, Q, R, a)
			sp = LiteState(belief, s.hiddenVal)
		else  # USE EKF
			belief = filter(ssm, obs, stateDistrib, Q, R, a)
			sp = LiteState(belief, s.hiddenVal)
		end

		# --- Update Hidden Belief, New Hidden Belief --- #
		mdp.hiddenBelief = mdp.hiddenBeliefNew

		# assume hidden state is always at end
		updateDistrib = Normal(s.hiddenVal, cov(belief)[end, end])

		mdp.hiddenBeliefNew[1] = cdf(updateDistrib, mdp.hiddenDiscrete[1])

		for i = 2:length(mdp.hiddenDiscrete)
			mdp.hiddenBeliefNew[i] = cdf(updateDistrib, mdp.hiddenDiscrete[i]) - cdf(updateDistrib, mdp.hiddenDiscrete[i-1])
		end
		# println("Old Belief: ", mdp.hiddenBelief)
		# println("New Belief: ", mdp.hiddenBeliefNew)
	end
	# println("Current States: ", mean(sp.estimState))
	return sp
end

# obtain observation, sampling from distribution P(o | theta', x', a')
function observation(mdp::LiteMDP,s::LiteState,a::Array{Float64,1})

	observedMeans = mean(s.estimState)[1:ssm.states]
	# sample a theta from hiddenBelief, pass into queryState
	queryState = vcat(observedMeans, s.hiddenVal)

	# obtain observation from that state, having taken action a
	obs = ssm.h(queryState, a)
	return obs
end

# --- Define Rewards and Actions --- #


# if prob == "2D"
#   # Defining action space for input controls
#   immutable FFTActionSpace
#       lower::Float64
#       upper::Float64
#   end

#   ### Defines the action space as continuous in range of fRange
#   function POMDPs.actions(mdp::LiteMDP)
#       #take rand action within force bounds
#       return FFTActionSpace(-fRange, fRange)#, mdp.force_range, mdp.force_range]
#   end

#   #Define POMDPs actions with function
#   POMDPs.actions(mdp::LiteMDP, s::LiteState, as::Array{Float64,1}) = actions(mdp)

#   ### Define rand for using within the range defined in FFT
#   function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
#       diff = as.upper-as.lower
#       return diff*rand(rng, ssm.nu)+as.lower
#   end

if prob == "Car"
    # Define action space
    immutable FFTActionSpace
        lower::Array{Float64, 1}
        upper::Array{Float64, 1}
    end

    # Defines the action space as continuous in range of fRange
    function POMDPs.actions(mdp::LiteMDP)
        # take rand action within force bounds
        return FFTActionSpace(-fRange, fRange) # frange is [u1 range; u2 range]
    end

    # Define POMDPs actions with function
    POMDPs.actions(mdp::LiteMDP, s::LiteState, as::Array{Float64,1}) = actions(mdp)

    ### Define rand for using within the range defined in FFT
    function Base.rand(rng::AbstractRNG, as::FFTActionSpace)
        diff = as.upper-as.lower
        return diff.*rand(rng, ssm.nu) + as.lower
    end

# -- 1D Case --- #
elseif prob == "1D"
  function POMDPs.actions(mdp::LiteMDP)
      #take rand action within force bounds
      return collect([f] for f in mdp.force_range)
  end

  POMDPs.actions(mdp::LiteMDP, s::LiteState, as::Normal{Float64})  = actions(mdp)

  #to use default random rollout policy implement action sampling funct
  function POMDPs.rand(rng::AbstractRNG, action_space::Normal{Float64}, dummy=nothing)
     return rand(action_space)
  end
end

### Calculate the reward for the current state and action
# Qg defined in Setup.jl file
function POMDPs.reward(mdp::LiteMDP,s::LiteState,a::Array{Float64,1})
	# reward + reward bonus
	# reward = expectation over hidden variable
	# reward from before: r = sum(abs.(s.trueState)'*-Qg) + sum(abs.(a)'*-Rg)
	# return r

	# r = 0
	# loop over all possible values of theta
	# for i = 1:length(mdp.hiddenDiscrete)
	#   theta = mdp.hiddenDiscrete[i]
	#   full_state = mean(s.estimState)
	#   obs_state = full_state[1:ssm.states]
	#   r_s = sum(abs.(obs_state)'*-Qg) + sum(abs.(a)'*-Rg)
	#   r = r + mdp.hiddenBelief[i]*r_s
	# end

	obs_state = mean(s.estimState)[1:ssm.states]
    if prob == "Car"
        TrueState = mean(s.estimState)
        r = Car_reward(TrueState[:], a[:], TrackIdx[end])
    else
        r = sum(abs.(obs_state)'*-Qg) + sum(abs.(a)'*-Rg)
    end
	rb = 0
	for ds in mdp.discreteStates
		for i = 1:length(mdp.hiddenDiscrete)
			theta = mdp.hiddenDiscrete[i]
			b = mdp.hiddenBelief[i]
			b_prime = mdp.hiddenBeliefNew[i]
			prob_sp = mdp.prob_sp_th[string((ds,theta))]
			rb = rb + prob_sp*b*abs(b_prime-b)
		end
	end

	lambda = 0.5
	# println("Reward: ", r)
	# println("Reward Bonus: ", rb)

	# return r
	return (r + lambda*rb)

end



# --- Check Terminal Case --- #
function POMDPs.isterminal(mdp::LiteMDP, s::LiteState) # just for a leaf --> reach terminal state won't
    # for states where the trace of the covariance is really high stop looking --> diverging EKF/UKF
    if trace(cov(s.estimState)) > cov_thresh # add or for the estimates below thresh
      #@show "out"
      return true
    end
    return false
end

# --- Constructor for Discount --- #

POMDPs.discount(mdp::LiteMDP) = mdp.discount_factor  # not even used?


# --- Generate Next State and Reward --- #

function POMDPs.generate_sr(mdp::LiteMDP, s::LiteState, a::Array{Float64,1}, rng::AbstractRNG)

    sp = transition(mdp,s,a)
    r = reward(mdp,s,a,sp)
    return sp, r
end

# --- Define Action --- #

# this function is called at every time step of outer simulation
