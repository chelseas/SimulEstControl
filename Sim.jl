# main simulation file that calls all other functions --> can run in parallel: julia -p X Sim.jl
@everywhere begin
  # cd to absolute path -->
  dir = pwd()
  cd(dir)

  # all parameter variables, packages, etc are defined here
  settings_file = "none" # name of data file to load
  settings_folder = "settings" # store data files here
  include("Setup.jl")

  ### processNoise and paramNoise pairs to be fed into numtrials worth simulations each
  function evaluating(params)
    totrew = 0.0 # summing all rewards with this
    if cross_entropy
        processNoise = params[6]
        paramNoise = params[7]
        alpha_act = 1.0/10.0 # alpha for action
        alpha_st = 1.0/20.0 # alpha for state
        k_act = max(1,floor(params[2]))/(n_iters^alpha_act) # k for action
        k_st = max(1,floor(params[1]))/(n_iters^alpha_st) # k for state
        solverCE = DPWSolver(n_iterations = Int(params[5]), depth = Int(max(1,floor(params[3]))), exploration_constant = params[4],
        k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)
        policyCE = solve(solverCE,mdp)
    else
        processNoise = params[1]
        paramNoise = params[2] # second element of tuple
    end

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
    step = 1 # count steps
    ### outer loop running for each simulation of the system
    for j = 1:numtrials # number of simulation trials run
        # Initialize saving variables between each run
        obs = zeros(ssm.ny,nSamples) #measurement history
        u = Array{Float64,2}(ssm.nu,nSamples) #input history
        x = Array{Float64,2}(ssm.nx,nSamples+1) #state history
        est = Array{Float64,2}(ssm.nx,nSamples+1) #store mean of state estimate
        uncertainty = Array{Float64,2}(ssm.nx*ssm.nx,nSamples+1) #store covariance of state estimate
        rewrun = Array{Float64,1}(nSamples) # total reward summed for each run
        act_dep_bounds = Array{Float64,1}(nSamples) # bounds chosen based on the taken action
        state_bounds = Array{Float64,1}(nSamples) # storing the bounds computed for each next state

        # initialize the state, belief, and stored values
        if fullobs
          xNew = MvNormal(x0_state,paramCov) # initialize the belief to exact state, param covariance because 0's throws error
        else
          xNew = MvNormal(est_list[:,j],paramCov) # initialize the belief state to wrong values
        end
        x[:,1] = x0_state # set actual initial state # make random? zach
        uncertainty[:,1] = reshape(cov(xNew),ssm.nx*ssm.nx) #store covariance
        est[:,1] = mean(xNew) # store average values of state
        if print_iters
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
                u[:,i] = MPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
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
            rewrun[i] = -sum(abs.(x[1:ssm.states,i])'*Qr) + -sum(abs.(u[:,i])'*Rg) # sum rewards

            # bounds testing
            #@show state_temp = x[1:ssm.states,i] # first 6 "measured" values
            #@show est_temp = MvNormal(mean(xNew)[ssm.states+1:end],cov(xNew)[ssm.states+1:end,ssm.states+1:end]) # MvNormal of Ests

            if bounds # show the state_bounds and see if they are within the threshold
              @show state_bounds[i] = norm(x[:,i+1])
              @show act_dep_bounds[i] = overall_bounds([-100.0],xNew,u[:,i],w_bound) # setting state_temp = [-100.0] to just use belief
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
    end
    [params,totrew/numtrials] # place variable here to have it output by evals
  end
end #@everywhere

# where the pmap function actually gets called
for k = 1:CE_iters
    evals = pmap(evaluating,pmapInput)#,paramNoiseList)
    if cross_entropy
        evals
        sorted = sortperm([evals[i][2] for i in 1:num_pop],rev=true)
        elite = sorted[1:num_elite]
        elite_params = evals[elite] # 1 x num_elite of arrays
        data_distrib = zeros(num_elite, CE_params)
        for e in 1:num_elite # store elite data
            data_distrib[e,:] = [elite_params[e][1][j] for j in 1:CE_params]
        end # data_distrib is num_elite x CE_num
        #data_distrib = convert(Array{Float64,2},data_distrib)
        try
            distrib = fit(typeof(CEset),data_distrib')
        catch ex
            if ex isa Base.LinAlg.PosDefException
                println("pos def exception")
                #distrib = fit(typeof(CEset), data_distrib'+= 0.01*randn(size(data_distrib)))
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
            cd("..")
        end

        if k != CE_iters # update CEset and pmapInput
            CEset = distrib
            pmapInput = []
            for i in 1:num_pop
                temp_CE = rand(CEset)
                push!(pmapInput,(temp_CE[1],temp_CE[2],temp_CE[3],temp_CE[4],n_iters,processNoiseList[1],paramNoiseList[1],sim_save_name,i,k+1))
            end
        end
    end
end
