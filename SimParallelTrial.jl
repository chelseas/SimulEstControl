# main simulation file that calls all other functions --> can run in parallel: julia -p X Sim.jl
@everywhere begin
  # cd to absolute path -->
  dir = pwd()
  cd(dir)

  # all parameter variables, packages, etc are defined here
  settings_file = "none" # name of data file to load
  settings_folder = "settings" # store data files here
  include("Setup.jl")

#=
  function initParams(x0_est,num_trial)
    srand(12+k_iter)
    est_list = rand(x0_est,numtrials) # pick random values around the actual state based on paramNoise for start of each trial
    return est_list
  end
=#
  ### processNoise and paramNoise pairs to be fed into numtrials worth simulations each
  function evaluating(params)
    totrew = 0.0 # summing all rewards with this

    processNoise = params[1]
    paramNoise = params[2] # second element of tuple
    k_iter = 1
    # Initializing an array of psuedo-random start states and actual state
    #srand(13) # seeding the est_list values so they will all be the same
    paramCov = paramNoise*eye(ssm.nx,ssm.nx) # covariance from paramNoise
    x0_est = MvNormal(state_init*ones(ssm.nx),paramCov) # initial belief
    #est_list = initParams(x0_est,k_iter)
    est_list = params[4] # just 1 vector, not a list

    x0_state = state_init*ones(ssm.nx) # actual initial state

    Q = diagm(processNoise*ones(ssm.nx))
    R = diagm(measNoise*ones(ssm.ny))
    w = MvNormal(zeros(ssm.nx),Q) # process noise distribution
    v = MvNormal(zeros(ssm.ny),measNoise*eye(ssm.ny,ssm.ny)) #measurement noise distribution
    step = 1 # count steps

    j = params[3]
    # Initialize saving variables between each run
    obs = zeros(ssm.ny,nSamples) #measurement history
    u = Array{Float64,2}(ssm.nu,nSamples) #input history
    x = Array{Float64,2}(ssm.nx,nSamples+1) #state history
    est = Array{Float64,2}(ssm.nx,nSamples+1) #store mean of state estimate
    uncertainty = Array{Float64,2}(ssm.nx*ssm.nx,nSamples+1) #store covariance of state estimate
    rewrun = Array{Float64,1}(nSamples) # total reward summed for each run

    # initialize the state, belief, and stored values
    if fullobs
      xNew = MvNormal(x0_state,paramCov) # initialize the belief to exact state, param covariance because 0's throws error
    else
      xNew = MvNormal(est_list,paramCov) # initialize the belief state to wrong values
    end
    x[:,1] = x0_state # set actual initial state # make random? zach
    uncertainty[:,1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance
    est[:,1] = mean(xNew) # store average values of state
    if print_iters || print_trials
        @show j # print the simulation trial number
    end

    ### inner loop running for each step in the simulation
    for i = 1:nSamples #for all samples
        if print_iters
            @show step = i # use to break out of some cases in the POMDP function
        end
        if printing @show i end
        cov_check = trace(cov(xNew))
        if cov_check > cov_thresh # input to action is exploding state
          u[:,i] = zeros(ssm.nu) # return action of zeros because unstable
          if print_iters
              @show "COV THRESH"
          end
        else # compute actions as normal for safe states
          if sim == "mcts"
            if cross_entropy
                u[:,i] = action(policyCE,xNew) # take an action MCTS
            else
                u[:,i] = action(policy,xNew) # take an action MCTS
            end
            if tree_vis # visualize the MCTS tree for 1 run and break
              inchrome(D3Tree(policy, title="whatever"))#,xNew)
              @show "Plotting Tree"
              break
            end
          elseif sim == "qmdp"
            AugNew = AugState(xNew)
            u[:,i] = action(policy, AugNew)
          elseif sim == "mpc"
              if reward_type == "region"
                  u[:,i] = MPCActionConstrained(xNew,nSamples+2-i,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
              else
                  u[:,i] = MPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
              end
          elseif sim == "smpc"
            u[:,i] = SMPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
          elseif sim == "snmpc"
            u[:,i] = SNMPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
          end
        end
        u[:,i] = control_check(u[:,i], x[:,i], debug_bounds) # bounding control
        x[:,i+1] = ssm.f(x[:,i],u[:,i]) + rand(w) # propagating the state
        if param_change # add additional term to the unknown params
          if param_type == "sine"
            x[:,i+1] = x[:,i+1] + [zeros(ssm.states); param_magn*cos(param_freq*i)*ones(ssm.nx - ssm.states)]
          elseif param_type == "steps"
            if i < 10
              param_var = 0
            elseif i < 20
              param_var = param_magn
            elseif i < 30
              param_var = 0
            elseif i < 40
              param_var = -param_magn
            else
              param_var = 0
            end
            x[:,i+1] = [x[1:ssm.states,i+1]; (state_init+param_var)*ones(ssm.nx - ssm.states)] + rand(w)
          end
        end
        x[:,i+1] = state_check(x[:,i+1], debug_bounds) # reality check --> see if values of parameters have gotten too small --> limit

        if reward_type == "L1"
            rewrun[i] = -sum(abs.(x[1:ssm.states,i])'*Qr) + -sum(abs.(u[:,i])'*Rg) # sum rewards
        elseif reward_type == "region"
            ms = x[1:ssm.states,i]
            if (ms[4] > region_lb[1]) && (ms[5] > region_lb[2]) && (ms[6] > region_lb[3]) && (ms[4] < region_ub[1]) && (ms[5] < region_ub[2]) && (ms[6] < region_ub[3])
                rewrun[i] = rew_in_region
                #@show "here"
            else
                rewrun[i] = rew_out_region
            end
            #@show x[1:ssm.states,i]
            #@show rewrun[i]
        end

        # bounds testing
        #@show state_temp = x[1:ssm.states,i] # first 6 "measured" values
        #@show est_temp = MvNormal(mean(xNew)[ssm.states+1:end],cov(xNew)[ssm.states+1:end,ssm.states+1:end]) # MvNormal of Ests

        if bounds_print # show the state_bounds and see if they are within the threshold
          @show state_bounds[(j-1)*nSamples+i] = norm(x[1:6,i+1]) # actual bounds for next state
          #@show act_dep_bounds[(j-1)*nSamples+i] = overall_bounds([-100.0],xNew,u[:,i],w_bound) # setting state_temp = [-100.0] to just use belief
        end

        if !fullobs # if system isn't fully observable update the belief
          # take observation of the new state
          obs[:,i] = ssm.h(x[:,i],u[:,i]) #+ rand(v) #<-- no measurement noise

          # update belief with current measurment, input
          if cov_check < cov_thresh # then update belief because we trust it
              xNew = ukf(ssm,obs[:,i],xNew,cov(w),cov(v),u[:,i]) # for UKF
          end

          # reality check --> see if estimates have gotten too extreme --> limit
          x_temp = mean(xNew)
          mean(xNew)[:] = est_check(x_temp, debug_bounds)
          est[:,i+1] = mean(xNew) #store mean belief
          uncertainty[:,i+1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance

          if printing
            @show u[:,i]
            @show x[:,i+1]
            @show mean(xNew)
            @show sum(sum(uncertainty[:,i+1]))
          end

        else # make xNew equivalent to exact state
          xNew = MvNormal(x[:,i+1],eye(ssm.nx,ssm.nx)) # exact update no covariance --> using eye cuz zeros not PSD
          est[:,i+1] = x[:,i+1]
        end
    end
    if print_iters
        @show mean(rewrun)
    end
    if save_best
        rew_best[j] = mean(rewrun)
    end
    totrew += mean(rewrun)
    if saving
      if cross_entropy
          parallel_num = [string(params[end-1])]
          sim_names = [string(params[end],sim_save_name),prob,sim,rollout,string(processNoise),string(paramNoise),string(numtrials),string(j)]
      else
          parallel_num = [""]
          sim_names = [sim_save_name,prob,sim,rollout,string(processNoise),string(paramNoise),string(numtrials),string(j)]
      end
      save_simulation_data(x,est,u,[rewrun rewrun]',uncertainty,prob_params,sim_names,parallel_num)
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

    gc() # clear data
    [params,totrew]
  end
end #@everywhere

# where the pmap function actually gets called
evals = pmap(evaluating,pmapInput)#,paramNoiseList)
