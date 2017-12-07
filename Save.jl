# ISSUE: currently cant rename the columns of dataframe so they are just random vars
using DataFrames
function save_data(data::Array, head::Vector, simsettings::Vector, data_val::String)
  # data_val here is what kind of data --> pos, est, unk, ctrl, rew, unc
  if simsettings[2]== "mcts"
    (prob,sim,rollout,processNoise,paramNoise,numtrails,run_num) = simsettings
    fname = string(data_val," ",prob," ",sim," ",rollout," PN ",processNoise," VARN ",paramNoise," Trial ",run_num,".csv")
  elseif simsettings[2]== "mpc"
    (prob,sim,rollout,processNoise,paramNoise,numtrials,run_num) = simsettings
    fname = string(data_val," ",prob," ",sim," ",rollout, " PN ",processNoise," VARN ",paramNoise," Trial ",run_num,".csv")
  end
  df = convert(DataFrame,data')
  #rename!(df, [head[i] for i in 1:length(head)])
  #@show df
  writetable(fname,df)
end

#=
# test example
a = ["1D","mpc","0.1","1"] # sim params
c = ["x","y","z","w"] # header
b = rand(4,100) # data
save_data(b,c,a,"rand")
=#

# Matrices of data, state names in hs, and settings
function save_simulation_data(s::Matrix, est::Matrix, ctrl::Matrix, rew::Array,
                              unc::Matrix, hs::Vector, settings::Vector,pname::Array)
  # get simulation properties from settings
  (ssn,prob,sim,rollout,processNoise,paramNoise,numtrials,run_num) = settings
  indset = settings[2:end] # to pass to each save_data command
  # Make folder for simulation
  try mkdir(data_folder)
  end
  cd(data_folder)
  try mkdir(ssn)
  end
  cd(ssn)
  newFolder = string(pname[1],prob," ",sim, " ",rollout," PN ",processNoise," VARN ",paramNoise," RUNS ",numtrials)
  try mkdir(newFolder)
  end
  cd(newFolder) # go into new folder to save files
  # Store data for states, est, control, reward, and unc
  save_data(s,hs,indset,"states")
  save_data(est,hs,indset,"est")
  save_data(ctrl,hs,indset,"ctrl")
  save_data(rew,hs,indset,"rew")
  save_data(unc,hs,indset,"unc")
  cd("../../..") # go back to main folder
end
