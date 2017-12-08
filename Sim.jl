@everywhere begin
  # cd to absolute path -->
    dir = pwd()
    cd(dir)
    #@show ARGS
    #@show length(ARGS)

    # SIM SETTINGS
    prob = "2D" # set to the "1D" or "2D" problems defined
    sim = "mcts" # mcts, mpc, qmdp, drqn
    rollout = "random" # MCTS/QMDP: random/position, DRQN: train/test
    bounds = false # set bounds for mcts solver
    quick_run = false
    numtrials = 10 # number of simulation runs
    noiseList = []
    cond1 = "full"

    #NOISE SETTINGS
    processNoiseList = [0.033]#[0.033, 0.1]#[0.001,0.0033,0.01,0.033,0.1,0.33] # default to full
    paramNoiseList = [0.1,0.3]#,0.5,0.7]
    ukf_flag = true # use ukf as the update method when computing mcts predictions
    param_change = false # add a cosine term to the unknown param updates
    param_type = "none" # sine or steps
    param_magn = 0.2 # magnitude of cosine additive term # use >0.6 for steps
    param_freq = 0.3

    # Output settings
    printing = false # set to true to print simple information
    plotting = false # set to true to output plots of the data
    saving = true # set to true to save simulation data to a folder # MCTS trial at ~500 iters is 6 min ea, 1hr for 10
    tree_vis = false # visual MCTS tree
    sim_save = "CE1" # name appended to sim settings for simulation folder to store data from runs
    data_folder = "dataCE2"
    fullobs = true # set to false for mpc without full obs
    if sim != "mpc" # set fullobs false for any other sim
      fullobs = false
    end

    # CROSS ENTROPY SETTINGS
    cross_entropy = true
    num_pop = 50 # number of samples to test this round of CE
    num_elite = 10 # number of elite samples to keep to form next distribution
    sim_save_name = string(sim_save,"_",prob,"_",sim,"_",cond1,"_",param_type,"_",fullobs)
    if cross_entropy
        niters_lb = 2490
        niters_ub = 2510
        states_lb = 1
        states_ub = 10
        act_lb = 10
        act_ub = 40
        depth_lb = 5
        depth_ub = 30
        expl_lb = 0.1
        expl_ub = 100.0
        CE_settings = [niters_lb,niters_ub,states_lb,states_ub,act_lb,act_ub,depth_lb,depth_ub,expl_lb,expl_ub]
        pmapInput = []
        for i in 1:num_pop
            push!(pmapInput,(rand(niters_lb:niters_ub),rand(states_lb:states_ub),rand(act_lb:act_ub),rand(depth_lb:depth_ub),rand(expl_lb:expl_ub),processNoiseList[1],paramNoiseList[1],sim_save_name,i))
        end
        @show pmapInput
        try mkdir(data_folder)
        end
        cd(data_folder)
        open(string(sim_save,".txt"), "w") do f
            write(f,string("Sim save: ",sim_save,"\n"))
            write(f,string("CE settings: ",CE_settings,"\n"))
            write(f,string("PMAP input: ",pmapInput,"\n"))
            cd("..")
        end
    else
        # combine the total name for saving
        for PRN in processNoiseList
            for PMN in paramNoiseList
                push!(noiseList,(PRN,PMN))
            end
        end
        pmapInput = noiseList
    end
    # all parameter variables, packages, etc are defined here
    include("Setup.jl")

  ### processNoise and paramNoise pairs to be fed into numtrials worth simulations each
  #for noise_setting = 1:length(paramNoiseList)
  function evaluating(params)#,processNoise::Float64)
    #processNoise = processNoiseList[noise_setting]
    #paramNoise = paramNoiseList[noise_setting]
    totrew = 0.0 # summing all rewards with this
    if cross_entropy
        processNoise = params[6]
        paramNoise = params[7]
        alpha_act = 1.0/10.0 # alpha for action
        alpha_st = 1.0/20.0 # alpha for state
        k_act = params[3]/(n_iters^alpha_act) # k for action
        k_st = params[2]/(n_iters^alpha_st) # k for state
        solverCE = DPWSolver(n_iterations = params[1], depth = params[4], exploration_constant = params[5],
        k_action = k_act, alpha_action = alpha_act, k_state = k_st, alpha_state = alpha_st)
        policyCE = solve(solverCE,mdp)
    else
        processNoise = params[1]
        paramNoise = params[2] # second element of tuple
    end
    # Initializing an array of psuedo-random start states and actual state
    #srand(13) # seeding the est_list values so they will all be the same
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
    @time for j = 1:numtrials # number of simulation trials run
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
        @show j # print the simulation trial number
        ### inner loop running for each step in the simulation
        @time for i = 1:nSamples #for all samples
            @show step = i # use to break out of some cases in the POMDP function
            if printing @show i end
            cov_check = trace(cov(xNew))
            if cov_check > cov_thresh # input to action is exploding state
              u[:,i] = zeros(ssm.nu) # return action of zeros because unstable
              @show "COV THRESH"
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
              elseif sim == "mpc"
                u[:,i] = MPCAction(xNew,nSamples+2-i)#n) # take an action MPC (n: # length of prediction horizon)
              elseif sim == "drqn"
                u[:,i] = MPCAction(xNew,nSamples+2-i)
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
              # take actual dynamics into ukf for oracle (deal with this later)
              #if ukf_flag
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
        @show mean(rewrun)
        totrew += mean(rewrun)
        # for each trial save simulation data
        if saving
          if cross_entropy
              parallel_num = [string(params[end])]
          else
              parallel_num = [""]
          end
          sim_names = [sim_save_name,prob,sim,rollout,string(processNoise),string(paramNoise),string(numtrials),string(j)]
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
        gc() # clear data?
    end
    [params,totrew/numtrials] # place variable here to have it output by evals
  end
end #@everywhere
evals = pmap(evaluating,pmapInput)#,paramNoiseList)
if cross_entropy
    @show evals
    @show sorted = sortperm([evals[i][2] for i in 1:num_pop],rev=true)
    @show elite = sorted[1:num_elite]
    @show elite_params = evals[elite]
    distrib = []
    for i in 1:5 # number of params that need to compute a range for
        data_distr = [elite_params[e][1][i] for e in 1:num_elite]
        push!(distrib,(maximum(data_distr),minimum(data_distr)))
    end
    @show distrib
    # Write out txt file with results of the CE round
    try mkdir(data_folder)
    end
    cd(data_folder)
    @show readdir()
    open(string(sim_save,".txt"), "w") do f
        write(f,string("Distrib: ",distrib,"\n"))
        write(f,string("Elite Params: ",elite_params,"\n"))
        write(f,string("Elite: ",elite,"\n"))
        write(f,string("Evals: ",evals,"\n"))
        cd("..")
    end
end
#end
# this prints, plots, and saves data
#include(Outputs.jl)
