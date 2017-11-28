using DataFrames
folder = "2D_0.7_step"#"test2"# give data folder name to compute averages from

# assuming you have all the
data_type = ["ctrl","est","rew","states","unc"] # all the first words of csv files to loop through

cd("data") # go into data folder
cd(folder)

# go into all of these folders and compute average values and std devs of the trials
sim_cases = readdir() # list of filenames in this directory
@show sim_cases
#@show sim_cases
f1 = sim_cases[1]
runs = parse(Int64,split(f1)[end]) # number of runs
# CREATE AVERAGES AND STD OF DATA FOR PLOTTING
for i = 1:length(sim_cases) # go through each data folder to compute averages
  curFolder = sim_cases[i]
  runs = parse(Int64,split(curFolder)[end]) # number of runs
  @show runs
  cd(curFolder)
  dataFiles = readdir() # all names of .csv files in this directory
  #@show dataFiles
  #@show dataFiles
  #@show dataFiles
  #@show length(dataFiles)
  dfr = reshape(dataFiles, runs, length(data_type))
  #@show dfr[:,1]
  for j = 1:length(data_type) # for each file type
    #keywords = parse(dfr[1,j]) # words from title of first in trials
    sim_sets = split(dfr[1,j]) # data_type(1), prob(2), sim(3), roll/obs(4), PN(5), PN#(6), VN(7), VN#(8), Trial, T#
    gen = readtable(dfr[1,j]) # load first table to get the size information
    nvars, numSteps = size(gen)
    data = Array{Float64,3}(runs, nvars, numSteps)
    @show size(data)
    #std = Array{Float64,3}(runs, numSteps, nvars)
    for k = 1:runs # for each trial compute an average and std
      @show dfr[k,j]
      temp = readtable(dfr[k,j]) # load individual file
      @show size(temp)
      data[k,:,:] = Array(temp)
    end
    # save new avg and std values --> just stack next to each other
    # do this both for the trials vs iteration
    avg = mean(data,1)
    #@show size(avg)
    std = (var(data,1).^0.5)/runs # standard error of the mean
    #@show j
    #@show sum(std)
    # cat avg and std matrices together to make it easier to bring in to plots
    data = cat(2,avg[1,:,:], std[1,:,:]) # size of num trials x nvars*2
    #@show size(data)
    # convert both to dataFrames and save in a new folder called processed data
    df = convert(DataFrame, data) # should this be transposed?
    #append!(a,b) to add words on to the end of the list to recreate a title
    # join(a) to merge list of words into string
    # data_type(1), prob(2), sim(3), roll/obs(4), PN(5), PN#(6), VN(7), VN#(8), Trial, T#
    fname_temp = join(sim_sets[1:8], " ") # new name for all but trial and T#
    #@show fname_temp
    fname = join([fname_temp ".csv"])
    #@show fname
    cd("../..")
    pdata = " processed data"
    newF = join([folder,pdata])
    try mkdir(newF)
    end
    cd(newF) # go into processed data file
    # STORE THE FILES here
    writetable(fname,df)

    # and for reward just overall trail sum (avg, sum, processN, paramN)
    if j == 3 # corresponding to the reward
      tot_dir = "total rewards"
      try mkdir(tot_dir)
      end
      cd(tot_dir)
      avgtot = mean(data,[1,3])[1,1,1] # return just the value for the mean
      stdtot = (var(mean(data,3),1)[1,1,1].^0.5)/runs # sum data along all numSteps and take std among runs
      PN = parse(Float64,sim_sets[6])
      PRN = parse(Float64,sim_sets[8])
      rew_ret = [avgtot stdtot PN PRN]' # returning the point to plot
      df2 = convert(DataFrame, [rew_ret rew_ret]')
      fname2 = join(["tot " fname])
      writetable(fname2,df2)
      cd("..")
    end

    cd("..")
    cd(folder)
    cd(curFolder)
  end

  cd("..") # back out of this folder
end

cd("../..") # change directory back to main folder

# Call PGFPlotting.jl here if I wnat
#include("PGFPlotting.jl")
