# Take the data from ParseData and make a variety of plots --> these will be auto passed
folder = "test2"
pdata = " processed data"
tot_dir = "total rewards"
plot_folder = "plots"
using PGFPlots
using DataFrames
pushPGFPlotsPreamble("\\usepackage{xfrac}")

# make sure all data for the specific plot you want is in the processed data folder specified
# For profile: make sure only plots are for 1 setting of Process and param Noise
# For varying: make sure only the 1 fixed variable is found in this folder

# run this and identify which of the 3 plot types you would like to make
# set this to determine what plots should be made
vary = false # plot varying process or param noise
varyMass = false # false if fixing mass and varying Process, true if varying mass
profile = true # plot the profile of the results

# general variables for plot labeling
mpc_leg = "MPC"
mcts_pos_leg = "MCTS Position"
mcts_rand_leg = "MCTS Random"
mcts_smooth_leg = "MCTS Smooth"
qmdp_leg = "QMDP"
mcts_pos_style = "red"
mcts_rand_style = "blue"
mpc_style = "green"
col4 = "purple"
col5 = "black"
mark1 = "none"

cd("data")
cd(join([folder pdata]))

# total reward plot
cd(tot_dir)
if vary
  tot_files = readdir() # all the files in tot_dir
  # loop through files and construct an array of the points and stdError to plot
  mcts_pos = Float64[]
  mcts_rand = Float64[]
  mpc = Float64[]
  tempName = split(tot_files[1])
  for i = 1:length(tot_files)
    curFile = tot_files[i]
    curData = readtable(curFile)
    curVec = Array(curData[1,:]) # array that I can use
    curName = split(curFile) # get words
    # initialize the arrays that we will plot
    sim = curName[4]
    rollout = curName[5]
    if sim == "mcts" # add to the array of mcts to plot
      if rollout == "position"
        mcts_pos = vcat(mcts_pos,curVec)
        #@show size(mcts_pos)
      elseif rollout == "random"
        mcts_rand = vcat(mcts_rand,curVec)
      end
    elseif sim == "mpc"
      mpc = vcat(mpc, curVec)
    end
    #@show mcts_pos
  end
  # go to plots directory
  cd("../..") # in data
  try mkdir(plot_folder)
  end
  cd(plot_folder) # in plots
  xaxis = L"Process Noise ($\sigma$)"
  yaxis = "Total Reward"
  vertspace = "0.0"
  horizspace = "0.5"
  if varyMass
    vv = 4
    title_start = "PRMV" # Param Varying start of plot name
  else
    vv = 3 # make 3 to plot for varying process, set to 4 for varyingParamNoise
    title_start = "PV" # Process varying start of plot name
  end
    #@show mcts_pos[:,vv]
  varyProcessPlot = Axis([ # comment out lines below if that sim type is not going to be plotted
      Plots.Linear(mcts_pos[:,vv],mcts_pos[:,1], errorBars = ErrorBars(y=mcts_pos[:,2]),style=mcts_pos_style,  mark=mark1, legendentry = mcts_pos_leg)
      #Plots.Linear(mcts_rand[:,vv],mcts_rand[:,1], errorBars = ErrorBars(y=mcts_rand[:,2]),style=mcts_pos_style,  mark=mark1, legendentry = mcts_rand_leg)
      #Plots.Linear(mcts_rand[:,vv],mcts_rand[:,1], errorBars = ErrorBars(y=mcts_rand[:,2]),style=mcts_pos_style,  mark=mark1, legendentry = mcts_rand_leg)
      #Plots.Linear(mcts_smooth[:,vv],mcts_smooth[:,1], errorBars = ErrorBars(y=mcts_smooth[:,2]),style=col4,  mark=mark1, legendentry = mcts_smooth_leg)
      #Plots.Linear(qmdp[:,vv],qmdp[:,vv], errorBars = ErrorBars(y=qmdp[:,vv]),style=col5,  mark=mark1, legendentry = qmdp_leg)
      ],xmode="log",xlabel = xaxis,ylabel=yaxis,legendPos="south west")
  title1 = join([title_start," TR ", tempName[3], " PN ", tempName[7], " VARN ", tempName[9][1:end-4]])
  save(string(title1,".pdf"),varyProcessPlot)
  save(string(title1,".svg"),varyProcessPlot)
  save(string(title1,".tex"),varyProcessPlot, include_preamble=false)
end
cd("../..") # in data
cd(join([folder pdata]))# back in test processed data folder

function format_data(A::Matrix)
  A1 = mean(abs.(A[:,1:Int(size(A)[2]/2)]),2)
  A2 = mean(abs.(A[:,Int(size(A)[2]/2)+1:end]),2)
  Afinal = [A1 A2]
  return Afinal
end

if profile # plot the profiles for the runs
  # avg vs nSamples plots
  files = readdir() # all average files
  filter!(x->x≠"total rewards",files) # remove total rewards folder
  # define all variables for plotting here so that they can be used outside the for loop
  tempName = split(files[1])
  #@show tempName
  offset = Int(length(files)/5) # dividing by 5 since 5 types of data for each sim
  if tempName[2] == "1D"
    states = 2
  elseif tempName[2] == "2D"
    states = 6 # last index of controllabel states for this problem
  end
  mcts_pos_st = Float64[]
  mcts_pos_est = Float64[]
  mcts_pos_ctrl = Float64[]
  mcts_pos_rew = Float64[]
  # add the variables for other simtypes
  for i = 1:offset
    ctrlCur = Array(readtable(files[i])) # control effort # nSamples x 2*dimensions
    estCur = Array(readtable(files[i+offset])) # estimates
    rewCur = Array(readtable(files[i+2*offset])) # rewCur
    sCur = Array(readtable(files[i+3*offset])) # states
    difCur = estCur - sCur # take difference of est and current
    rewCur = rewCur[:,[1,3]]
    difS = sCur[:,1:states] # don't subtract state values --> want to see go to zero
    difE = difCur[:,1+states:end]
    difForm = format_data(difCur)
    curName = split(files[i]) # get words from conrol file
    # initialize the arrays that we will plot
    sim = curName[3]
    rollout = curName[4]
    if sim == "mcts" # want to plot the
      if rollout == "position"
        mcts_pos_st = format_data(difS) # all of these are nSamples/nSamples+1 x 2 (avg and std)
        mcts_pos_est = format_data(difE)
        mcts_pos_ctrl = format_data(ctrlCur)
        mcts_pos_rew = rewCur
        #@show size(mcts_pos_ctrl)
        #@show size(mcts_pos_ctrl)
      elseif rollout == "random"
      end
    elseif sim == "mpc"
    end
  end
  #### Plot all 4 on a subplots
  stlab = "States"
  estlab = "Parameters"
  ctrllab = "Control"
  rewlab = "Reward"
  xaxis = "Steps (0.1 seconds)"
  vertspace = "0.5"
  horizspace = "0.0"
  ht = "4cm"
  wdth = "8cm"
  nSamples = size(mcts_pos_ctrl)[1]
  st = Axis([
      Plots.Linear(0:nSamples-1,mcts_pos_st[1:nSamples,1],errorBars = ErrorBars(y=mcts_pos_st[1:nSamples,2]), style=mcts_pos_style,  mark=mark1, legendentry=mcts_pos_leg)
      Plots.Linear([0, nSamples-1],[0,0], style=col5,  mark=mark1)
      ],ylabel=stlab,legendPos="north east", width = wdth, height=ht)

  est = Axis([
  Plots.Linear(0:nSamples-1,mcts_pos_est[1:nSamples,1],errorBars = ErrorBars(y=mcts_pos_est[1:nSamples,2]), style=mcts_pos_style,  mark=mark1)
      Plots.Linear([0, nSamples-1],[0,0], style=col5,  mark=mark1)
      ], ylabel=estlab, width = wdth, height=ht) #"Velocity (m/s)"

  ctrl = Axis([
      #Plots.Linear(1:safeiters,LQG18[1:safeiters,var3+ex], style=mcts_pos_style,  mark=mark1)
      Plots.Linear(0:nSamples-1,mcts_pos_ctrl[:,1],errorBars = ErrorBars(y=mcts_pos_ctrl[:,2]), style=mcts_pos_style,  mark=mark1)
      Plots.Linear([0, nSamples-1],[0,0], style=col5,  mark=mark1)
      ], ylabel=ctrllab, width = wdth, height=ht)

  rew = Axis([
      Plots.Linear(0:nSamples-1,mcts_pos_rew[:,1],errorBars = ErrorBars(y=mcts_pos_rew[:,2]), style=mcts_pos_style,  mark=mark1)
      ],xlabel=xaxis, ylabel=rewlab, width = wdth, height = ht)

  g3 = GroupPlot(1,4, groupStyle = string("horizontal sep = ",horizspace,"cm, vertical sep = ",vertspace,"cm"))
  push!(g3, st)
  push!(g3, est)
  push!(g3, ctrl)
  push!(g3, rew)
  title_start = "PROF "
  title3 = join([title_start, tempName[2], " PN ", tempName[6], " VARN ", tempName[8][1:end-4]])

  ####
  cd("..")
  try mkdir(plot_folder)
  end
  cd(plot_folder)
  save(string(title3,".pdf"),g3)
  save(string(title3,".svg"),g3)
  save(string(title3,".tex"),g3, include_preamble=false)

end
cd("../..")
