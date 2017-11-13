# TODO:
# ask zach how I can sample from a linear distribution rather than Normal{Float64} in POMDPs.actions for 1D prob
# merge the POMDP dfeinition files for all problems
# create a modified MDP to include the previous action in the state to account for the smoothness term
# parallelize
# figure out why UKF is performing so poorly when used in MCTS exploration
# add another outer loop to specify the simulations to run for all the conditions including roll-outs
# is 1D MCTS sampling from a normal distriution for actions?
# check if using right 2D dynamics and if MPC is the same
# add the standard deviation lines as well and shade the region of it instead of error for profiles

# Tests to run:
# run EKF and UKF for a variety of process noises and do performance comp --> MPC then maybe mcts

# To use:
# cd("C:/Users/patty/Box Sync/SimulEstControl/SimulEstV0") # change to your path
# include("simulation.jl") # runs this file

# Specify simulation parameters
prob = "1D" # set to the "1D" or "2D" problems defined
sim = "qmdp"  # "qmdp"
rollout = "random"
quick_run = false
numtrials = 1 # number of simulation runs
processNoiseList = [0.001] #, 0.1]
paramNoiseList = [0.001] #, 10.0]
ukf_flag = true # use ukf as the update method when computing mcts predictions

# Output settings
printing = false # set to true to print simple information
plotting = true # set to true to output plots of the data
saving = true # set to true to save simulation data to a folder
sim_save_name = "10TrialTest" # name appended to sim settings for simulation folder to store data from runs
if sim == "mpc"
  fullobs = true # set to false for mpc without full obs
else
  fullobs = false
end

# all parameter variables, packages, etc are defined here
include("Setup.jl")
#=
sim_set = ["mpc", "mcts", "mcts"] # set to "mpc" or "mcts"
rollout_set = ["nobs","position","random"] # sets rollout policy for MCTS to "position", "random", or "smooth"
sim = sim_set[1]
rollout = rollout_set[1]

# run the different simulation conditions
for sim_setting = 1:length(sim_set)
  sim = sim_set[sim_setting]
  roll_ind = rollout_set[sim_setting]
  if sim == "mpc"
    if roll_ind == "obs"
      fullobs = true
    else
      fullobs = false
    end
  elseif sim == "mcts"
    fullobs = false
    rollout = rollout_ind
  end
  =#
  ### processNoise and paramNoise pairs to be fed into numtrials worth simulations each
  for noise_setting = 1:length(paramNoiseList)
    processNoise = processNoiseList[noise_setting]
    paramNoise = paramNoiseList[noise_setting]

    # Initializing an array of psuedo-random start states and actual state
    srand(13) # seeding the est_list values so they will all be the same
    paramCov = paramNoise*eye(ssm.nx,ssm.nx) # covariance from paramNoise
    x0_est = MvNormal(state_init*ones(ssm.nx),paramCov) # initial belief
    est_list = rand(x0_est,numtrials) # pick random values around the actual state based on paramNoise for start of each trial
    x0_state = state_init*ones(ssm.nx) # actual initial state

    Q = diagm(processNoise*ones(ssm.nx))
    R = diagm(measNoise*ones(ssm.ny))
    w = MvNormal(zeros(ssm.nx),Q) # process noise distribution
    v = MvNormal(zeros(ssm.ny),measNoise*eye(ssm.ny,ssm.ny)) #measurement noise distribution

    ### outer loop running for each simulation of the system
    @time for j = 1:numtrials # number of simulation trials run

        # Initialize saving variables between each run
        obs = zeros(ssm.ny,nSamples) #measurement history
        u = Array{Float64,2}(ssm.nu,nSamples) #input history
        x = Array{Float64,2}(ssm.nx,nSamples+1) #state history
        est = Array{Float64,2}(ssm.nx,nSamples+1) #store mean of state estimate
        uncertainty = Array{Float64,2}(ssm.nx*ssm.nx,nSamples+1) #store covariance of state estimate
        rewrun = Array{Float64,1}(nSamples) # total reward summed for each run

        # TO DO:
        # - work with AugState state
        # - call action(policy, AugState), do we actually need to create a new action fxn
        #   because AugState is different?

        # initialize the state, belief, and stored values
        if fullobs
          xNew = MvNormal(x0_state,paramCov) # initialize the belief to exact state, param covariance because 0's throws error
        else
          xNew = MvNormal(est_list[:,j],paramCov) # initialize the belief state to wrong values
        end
        x[:,1] = x0_state # set actual initial state # make random? zach
        uncertainty[:,1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance
        est[:,1] = mean(xNew) # store average values of state
        @show j # print the simulation trial number

        ### inner loop running for each step in the simulation
        @time for i = 1:nSamples #for all samples
            if printing @show i end
            if trace(cov(xNew)) > cov_thresh # input to action is exploding state
              u[:,i] = zeros(ssm.nu) # return action of zeros because unstable
              @show "COV THRESH INPUT EXCEEDED"
            else # compute actions as normal for safe states
              if sim == "mcts"
                u[:,i] = action(policy,xNew) # take an action MCTS
              elseif sim == "qmdp"
              	# EDIT THIS
                u[:, i] = action(policy, xNew)
              elseif sim == "mpc"
                u[:,i] = MPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
              end
            end
            u[:,i] = control_check(u[:,i], x[:,i], debug_bounds) # bounding controls
            x[:,i+1] = ssm.f(x[:,i],u[:,i]) + rand(w) # propagating the state
            x[:,i+1] = state_check(x[:,i+1], debug_bounds) # reality check --> see if values of parameters have gotten too small --> limit
            rewrun[i] = -sum(abs.(x[1:ssm.states,i])'*Qr) + -sum(abs.(u[:,i])'*Rg) # sum rewards

            if !fullobs # if system isn't fully observable update the belief
              # take observation of the new state
              obs[:,i] = ssm.h(x[:,i],u[:,i]) #+ rand(v) #<-- no measurement noise
              # update belief with current measurment, input
              # take actual dynamics into ukf for oracle (deal with this later)
              #if ukf_flag
                xNew = ukf(ssm,obs[:,i],xNew,cov(w),cov(v),u[:,i]) # for UKF
              #else
                #xNew = filter(ssm,obs[:,i],xNew,Q,R,u[:,i]) # for EKF
              #end


              # reality check --> see if estimates have gotten too extreme --> limit
              x_temp = mean(xNew)
              mean(xNew)[:] = est_check(x_temp, debug_bounds)
              est[:,i+1] = mean(xNew) #store mean belief
              uncertainty[:,i+1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance

              if printing
                @show u[:,i]
                @show x[:,i+1]
                @show mean(xNew)
                @show uncertainty[:,i+1]
              end

            else # make xNew equivalent to exact state
              xNew = MvNormal(x[:,i+1],eye(ssm.nx,ssm.nx)) # exact update no covariance --> using eye cuz zeros not PSD
              est[:,i+1] = x[:,i+1]
            end
        end

        # for each trial save simulation data
        if saving
          sim_names = [sim_save_name,prob,sim,rollout,string(processNoise),string(paramNoise),string(numtrials),string(j)]
          save_simulation_data(x,est,u,[rewrun rewrun]',uncertainty,prob_params,sim_names)
          # I'm putting two reward vectors to avoid vector error
        end
        if printing
          @show processNoise
          @show avgcost = sum(rewrun)/numtrials # avg tot rewards for the runs
          @show sum(u)/nSamples # avg control effort
          #@show est[3,:]#sum(est[3,:])/nSamples # all mass est
          @show est[:,nSamples+1] # last estimate
          @show x[:,nSamples+1] # last element of states
          #@show sum(uncertainty,1)
        end
        if plotting
          # for multiple runs these variables will have to be averaged for t!!!
          #include("Plot.jl")
          lwv = 2
          # Plot position states and estimates
          #pos_pl_data = vcat(x[startState:ssm.states,:],est[startState:ssm.states,:])
          pos_pl_data = x[startState:ssm.states,:]
          posest_pl_data = est[startState:ssm.states,:]

          pos_pl = plot(pos_pl_data', linewidth=2, title="Position")
          pos_est = plot(posest_pl_data', linewidth=2, title="Position Est")
          # Plot velocity states and estimates
          #vel_pl_data = vcat(x[1:startState-1,:],est[1:startState-1,:])
          vel_pl_data = x[1:startState-1,:]
          velest_pl_data = est[1:startState-1,:]

          vel_pl = plot(vel_pl_data', lw = lwv, title = "Velocity")
          vel_est = plot(velest_pl_data', lw = lwv, title = "Velocity Est")

          # Plot unknown param states and estimates
          #unk_pl_data = vcat(x[ssm.states+1:end,:],est[ssm.states+1:end,:])
          unk_pl_data = x[ssm.states+1:end,:]
          unkest_pl_data = est[ssm.states+1:end,:]
          unk_pl = plot(unk_pl_data', lw = lwv, title = "Unknown Params")
          unk_est = plot(unkest_pl_data', lw = lwv, title = "Unknown Params Est")

          # Plot control effort
          control_pl = plot(u', lw = lwv, title = "Control")
          # Plot rewards
          rew_pl = plot(rewrun, lw = lwv, title = "Reward")
          # Subplot all of them together
          #label = join([sim," ","Rew ",sum(rew_pl)," PN ", string(processNoise), " VARN ",string(paramNoise)])
          display(plot(pos_pl,pos_est,vel_pl,vel_est,unk_pl,unk_est,control_pl,rew_pl,layout=(4,2)))#,xlabel=label)
          #savefig(join(["test " string(j) ".png"])) # save plots for each run
        end
        gc() # clear data?
    end
  end
#end
# this prints, plots, and saves data
#include(Outputs.jl)
