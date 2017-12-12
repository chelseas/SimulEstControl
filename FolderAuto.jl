# Moves files from the general saved format into folders grouped by VARN
data_folder = "general"
new_data_folder = string(data_folder,"_mod")
varn = true # group into folders for given varn values

cd(data_folder)
sims = readdir() # should have a folder for each mpc, mcts, etc
for sim in sims
    cd(sim) # go into sim folder containing folders of all noise permutations
    conds = readdir()
    varns = []
    for cond in conds
        # get list of all the variation noise values
        temp = split(cond)
        push!(varns,temp[end-2]) # word corresponding to VARN value
    end
    # make all new directories
    cd("../..") # to main folder
    try mkdir(new_data_folder) # make new data folder and cd to it
    end
    cd(new_data_folder)
    for i in varns # make all new data folders
        try mkdir(string(i)) # make new data folder with varn as string
        end
        #push!(dest_dir,string("../../",new_data_folder,"/",i)) # destination directory
    end
    cd("..") # back to main folder
    cd(data_folder)
    cd(sim)
    # copy the right conds files to the appropriate dest_dir
    for cond in conds
        temp = split(cond)
        cur_var = temp[end-2]
        dest = string("../../",new_data_folder,"/",cur_var,"/",cond) # destination folder path
        cp(cond,dest) # copy cond folder to dest path --> do I need to add cond to dest path?
    end
    cd("..") # back to data folders
end
cd("..") # back to main folder
