# main simulation file that calls all other functions --> can run in parallel: julia -p X Sim.jl
@everywhere begin
  # cd to absolute path -->
  dir = pwd()
  cd(dir)

  # all parameter variables, packages, etc are defined here
  settings_file = "none"#mpc_unk_reg_depth10" # name of data file to load
  settings_folder = "set2" # store data files here
  include("Setup.jl")

  function initParams(x0_est,k_iter)
    srand(12+k_iter)
    est_list = rand(x0_est,numtrials) # pick random values around the actual state based on paramNoise for start of each trial
    return est_list
  end

  ### processNoise and paramNoise pairs to be fed into numtrials worth simulations each
  function evaluating(params)
    totrew = 0.0 # summing all rewards with this
    if cross_entropy
        processNoise = params[6]
        paramNoise = params[7]
        alpha_act = 1.0/30.0 # alpha for action
        alpha_st = 1.0/30.0 # alpha for state
        k_act = max(1,params[1])/(n_iters^alpha_act) # k for action
        k_st = max(1,params[4])/(n_iters^alpha_st) # k for state
        solverCE = DPWSolver(n_iterations = Int(params[5]), depth = Int(params[2]), exploration_constant = params[3],
        k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)
        policyCE = MCTS.solve(solverCE,mdp)
        k_iter = params[end]
        if save_best
            rew_best = zeros(numtrials)
        end
    else
        processNoise = params[1]
        paramNoise = params[2] # second element of tuple
        k_iter = 1
    end

    # Initializing an array of psuedo-random start states and actual state
    #srand(13) # seeding the est_list values so they will all be the same
    paramCov = paramNoise*eye(ssm.nx,ssm.nx) # covariance from paramNoise
    x0_est = MvNormal(state_init*ones(ssm.nx),paramCov) # initial belief
    est_list = initParams(x0_est,k_iter)

    if cross_entropy
        srand(params[end-1]) # random seed for each parallel measure --> just want the same initial params
    end

    #est_list = rand(x0_est,numtrials) # pick random values around the actual state based on paramNoise for start of each trial
    x0_state = state_init*ones(ssm.nx) # actual initial state

    Q = diagm(processNoise*ones(ssm.nx))
    R = diagm(measNoise*ones(ssm.ny))
    w = MvNormal(zeros(ssm.nx),Q) # process noise distribution
    v = MvNormal(zeros(ssm.ny),measNoise*eye(ssm.ny,ssm.ny)) #measurement noise distribution
    step = 1 # count steps
    ### outer loop running for each simulation of the system
    if bounds_print
        act_dep_bounds = Array{Float64,1}(nSamples*numtrials) # bounds chosen based on the taken action
        state_bounds = Array{Float64,1}(nSamples*numtrials) # storing the bounds computed for each next state
        #within_bounds_cnt = 1
        rews = zeros(numtrials)
    end
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
          xNew = MvNormal(x0_state,paramCov) # initialize the belief to exact state, param covariance because 0's throws error
        else
          xNew = MvNormal(est_list[:,j],paramCov) # initialize the belief state to wrong values
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
                  #break
                end
              elseif sim == "qmdp"
                AugNew = AugState(xNew)
                u[:,i] = action(policy, AugNew)
              elseif sim == "mpc"
                  if reward_type == "region"
                      u[:,i] = MPCActionConstrained(xNew,depths,depths)#nSamples+2-i,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
                  else
                      u[:,i] = MPCAction(xNew,depths)#nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
                  end
              elseif sim == "smpc"
                u[:,i] = SMPCAction(xNew,depths)#nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
              elseif sim == "snmpc"
                u[:,i] = SNMPCAction(xNew,depths)#nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
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
        if bounds_print
            rews[j] = mean(rewrun)
        end
    end# end j loop
    if bounds_print
        thresh_funct = x -> (x > desired_bounds)
        @show count(thresh_funct,state_bounds)/(numtrials*nSamples)
        #@show count(thresh_funct,act_dep_bounds)/(numtrials*nSamples)
        @show mean(rews)
        @show std(rews)
        if !bounds_save # running standard MCTS without bounds
            try mkdir(data_folder)
            end
            cd(data_folder)
            open(string(sim_save," ",processNoise," ",paramNoise," Bounds ", desired_bounds,".txt"), "w") do f
                thresh_funct = x -> (x > desired_bounds)
                out_of_bounds = count(thresh_funct,state_bounds)/(numtrials*nSamples)
                write(f,string("Percent out of bounds: ",out_of_bounds,"\n"))
                write(f,string("Desired Bounds: ",desired_bounds,"\n"))
                write(f,string("Params: ",processNoise," ",paramNoise,"\n"))
                write(f,string("Max Action Count: ",max_action_count,"\n"))
                write(f,string("Rew: ",mean(rews),"\n"))
                write(f,string("Sample STD: ", std(rews)/sqrt(numtrials),"\n"))
            end
            cd("..")
        end
    end
    if bounds_save
        try mkdir(data_folder)
        end
        cd(data_folder)
        open(string(sim_save," ",processNoise," ",paramNoise," Bounds ", desired_bounds,".txt"), "w") do f
            thresh_funct = x -> (x > desired_bounds)
            out_of_bounds = count(thresh_funct,state_bounds)/(numtrials*nSamples)
            out_of_bounds_possible = (count(thresh_funct,state_bounds) - action_limit_count)/(numtrials*nSamples - action_limit_count)
            write(f,string("Percent out of bounds: ",out_of_bounds,"\n"))
            write(f,string("Percent out of bounds for possible actions: ",out_of_bounds_possible,"\n"))
            write(f,string("Count for no safe actions: ",action_limit_count,"\n"))
            write(f,string("Desired Bounds: ",desired_bounds,"\n"))
            write(f,string("Params: ",processNoise," ",paramNoise,"\n"))
            write(f,string("Max Action Count: ",max_action_count,"\n"))
            write(f,string("Rew: ",mean(rews),"\n"))
            write(f,string("Sample STD: ", std(rews)/sqrt(numtrials),"\n"))
        end
        cd("..")
    end
    if save_best # compute avg and std, and maybe write
        # read in current best mean value
        cd(data_folder)
        try # try opening file if it exists
            g = open(string("best ",processNoise," ",paramNoise," ",sim_save,".txt")) # start reading doc
            lines = readlines(g)
            save_best_mean = parse(Float64,lines[1])
        catch
            save_best_mean = -10000 # no file so make it always reset the save_best_mean
        end
        cd("..")
        temp_avg = mean(rew_best)
        if temp_avg > save_best_mean # save this data
            save_best_mean = temp_avg
            save_best_std = std(rew_best)
            # write out to file
            cd(data_folder)
            open(string("best ",processNoise," ",paramNoise," ",sim_save,".txt"), "w") do f
                write(f,string(save_best_mean,"\n")) # avg
                write(f,string(save_best_std,"\n")) # std
                write(f,string("Settings: ",params,"\n"))
            end
            cd("..")
        end
    end
    if print_iters
        @show totrew/numtrials
    end
    [params,totrew/numtrials] # place variable here to have it output by evals
  end
end #@everywhere

# where the pmap function actually gets called
for k = 1:CE_iters
    evals = pmap(evaluating,pmapInput)#,paramNoiseList)
    if cross_entropy
        #evals
        sorted = sortperm([evals[i][2] for i in 1:num_pop],rev=true)
        elite = sorted[1:num_elite]
        elite_params = evals[elite] # 1 x num_elite of arrays
        data_distrib = zeros(num_elite, CE_params)
        for e in 1:num_elite # store elite data
            data_distrib[e,:] = [elite_params[e][1][j] for j in 1:CE_params] # 2:CE+1 because skipping the first value which is fixed as 1 for number of states sampled
        end # data_distrib is num_elite x CE_num
        #data_distrib = convert(Array{Float64,2},data_distrib)
        try
            distrib = fit(typeof(CEset),data_distrib')
        catch ex
            if ex isa Base.LinAlg.PosDefException
                println("Fit CE distribution has pos def tweaked")
                data_adj = data_distrib + 0.01*rand(size(data_distrib))
                distrib = fit(typeof(CEset), data_adj')
            else
                rethrow(ex)
            end
        end
        # Write out txt file with results of the CE round
        try mkdir(data_folder)
        end
        #sim_save = string(k," ",sim_save)
        cd(data_folder)
        open(string(k," ",sim_save,".txt"), "w") do f
            write(f,string("Sim save: ",k," ",sim_save,"\n"))
            write(f,string("CE settings: ",CEset,"\n"))
            write(f,string("PMAP input: ",pmapInput,"\n"))
            write(f,string("Distrib: ",distrib,"\n"))
            write(f,string("Elite Params: ",elite_params,"\n"))
            write(f,string("Elite: ",elite,"\n"))
            write(f,string("Evals: ",evals,"\n"))
        end
        open(string(sim_save,"_overall.txt"),"a") do g
            write(g,string(k,": ",distrib,"\n"))
        end
        cd("..")
        if k != CE_iters # update CEset and pmapInput
            CEset = distrib
            pmapInput = CE_sample(distrib,num_pop,n_iters,processNoiseList[1],paramNoiseList[1],sim_save_name,k+1)
        end
        max_eig = maximum(eigvals(cov(distrib))) # find max eigenvalue of sampled distrib
        if (saving == true) || ((max_eig < max_eig_cutoff) && !save_last) # i.e. the last round the max eig was less than threshold need to break out
            break
        end
        if ((k == CE_iters - 1) || (max_eig < max_eig_cutoff)) && save_last # about to start last CE round if last iter or max eig below threshold
            @everywhere saving = true
        end
    end
end
