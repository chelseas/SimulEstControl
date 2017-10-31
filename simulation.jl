# TODO:
# create a modified MDP to include the previous action in the state to account for the smoothness term
# see if passing in a state outside of the isTerminal allowance is causing problems for the DPW actions function?
# make a data parsing function which will take folder of sim data and compute avg and standard deviation for state,est,actions,rews
# fix PGF plots so that can make high quality figs

# cd("C:/Users/patty/Box Sync/SimulEstControl/SimulEstV0") # change to your path
# include("simulation.jl") # runs this file

# specify simulation parameters
prob = "1D" # set to the "1D" or "2D" problems defined
sim = "mcts" # set to "mpc" or "mcts"
rollout = "position" # sets rollout policy for MCTS to "position", "random", or "smooth"
fullobs = false # set to true if you want to use actual state params in place of the belief

# output settings
printing = true # set to true to print simple information
plotting = false # set to true to output plots of the data
saving = false # set to true to save simulation data to a folder
sim_save_name = "test" # name appended to sim settings for simulation folder to store data from runs
quick_run = false # set the simulation iterations to 10 for quickly debugging

# all parameter variables, packages, etc are defined here
include("Setup.jl")

### outer loop running for each simulation of the system
for j = 1:numtrials # number of simulation trials run

    # Initialize saving variables between each run
    obs = zeros(ssm.ny,nSamples) #measurement history
    u = Array{Float64,2}(ssm.nu,nSamples) #input history
    x = Array{Float64,2}(ssm.nx,nSamples+1) #state history
    est = Array{Float64,2}(ssm.nx,nSamples+1) #store mean of state estimate
    uncertainty = Array{Float64,2}(ssm.nx*ssm.nx,nSamples+1) #store covariance of state estimate
    rewrun = Array{Float64,1}(nSamples) # total reward summed for each run

    # initialize the state, belief, and stored values
    if fullobs
      xNew = MvNormal(x0_state,eye(ssm.nx,ssm.nx)) # initialize the belief to exact state
    else
      xNew = x0_est # initialize the belief state to wrong values
    end
    x[:,1] = x0_state # set actual initial state # make random? zach
    uncertainty[:,1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance
    est[:,1] = mean(xNew) # store average values of state
    @show j # print the simulation trial number

    ### inner loop running for each step in the simulation
    @time for i = 1:nSamples #for all samples
        @show i
        if sim == "mcts"
          u[:,i] = action(policy,xNew) # take an action MCTS
        elseif sim == "mpc"
          u[:,i] = MPCAction(xNew,n) # take an action MPC (n: # length of prediction horizon)
        end

        u[:,i] = control_check(u[:,i], x[:,i], debug_bounds) # bounding controls
        x[:,i+1] = ssm.f(x[:,i],u[:,i]) + rand(ssm.w) # propagating the state
        x[:,i+1] = state_check(x[:,i+1], debug_bounds) # reality check --> see if values of parameters have gotten too small --> limit
        rewrun[i] += sum(abs.(x[1:ssm.states,i])'*Qr) + sum(abs.(u[:,i])'*Rg) # sum rewards (added "."'s here )

        if !fullobs # if system isn't fully observable update the belief
          # take observation of the new state
          obs[:,i] = ssm.h(x[:,i],u[:,i]) #+ rand(v) #<-- no measurement noise
          # update belief with current measurment, input
          # take actual dynamics into ukf for oracle (deal with this later)
          #xNew = filter(ssm,obs[:,i],xNew,u[:,i]) # for EKF
          xNew = ukf(ssm,obs[:,i],xNew,cov(ssm.w),cov(ssm.v),u[:,i]) # for UKF

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
            if trace(cov(xNew)) > cov_thresh
              @show "COV THRESH INPUT EXCEEDED"
            end
          end

        else # make xNew equivalent to exact state
          xNew = MvNormal(x[:,i+1],eye(ssm.nx,ssm.nx)) # exact update no covariance --> using eye cuz zeros not PSD
        end
    end

    # for each trial save simulation data
    if saving
      sim_names = [sim_save_name,prob,sim,rollout,string(processNoise),string(j)]
      save_simulation_data(x,est,u,[rewrun rewrun],uncertainty,prob_params,sim_names)
      # I'm putting two reward vectors to avoid vector error
    end
    if printing
      # print results
      @show processNoise
      @show avgcost = sum(rewrun)/numtrials # avg tot rewards for the runs
      @show sum(u)/nSamples # avg control effort
      @show est[3,:]#sum(est[3,:])/nSamples # avg mass est
      @show est[:,nSamples+1] # last estimate
      @show x[:,nSamples+1] # last element of states
      @show sum(uncertainty,1)
      #
    end
    if plotting
      # for multiple runs these variables will have to be averaged for t!!!
      #include("Plot.jl")
      using Plots
      plotly()
      lwv = 2
      # Plot position states and estimates
      pos_pl_data = vcat(x[startState:ssm.states,:],est[startState:ssm.states,:])
      pos_pl = plot(pos_pl_data', linewidth=2, title="Position")
      # Plot velocity states and estimates
      vel_pl_data = vcat(x[1:startState-1,:],est[1:startState-1,:])
      vel_pl = plot(vel_pl_data, lw = lwv, title = "Velocity")
      # Plot unknown param states and estimates
      unk_pl_data = vcat(x[ssm.states+1:end,:],est[ssm.states+1:end,:])
      unk_pl = plot(unk_pl_data, lw = lwv, title = "Unknown Params")
      # Plot control effort
      control_pl = plot(u', lw = lwv, title = "Control")
      # Plot rewards
      rew_pl = plot(rewrun', lw = lwv, title = "Reward")
      # Subplot all of them together
      plot(pos_pl,vel_pl,unk_pl,control_pl,rew_pl,layout=(5,1))
    end
end

# this prints, plots, and saves data
#include(Outputs.jl)
