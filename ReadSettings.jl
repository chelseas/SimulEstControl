# example of reading in a setup file
function string_as_varname(s::AbstractString,v::Any)
         s=Symbol(s)
         @eval (($s) = ($v))
end

dir = pwd()
cd(dir)

if settings_file != "none"
    cd(settings_folder)
    f = open(string(settings_file,".txt")) # start reading doc
    lines = readlines(f)
    cnter = 0 # count for the strings, ints, and floats
    for line in lines
      temp = split(line) # break by spaces
      if (temp[1] == "#") || (length(temp) == 0) # don't process commented lines
          cnter += 1
      else # take first word as value, last word as
          val_temp = temp[1] # starts as string
          var_temp = temp[end] # always a string
          if val_temp == "true" || val_temp == "false"
              val_temp = parse(Bool,val_temp)
          elseif cnter == 2 # input is int
              val_temp = parse(Int64,val_temp)
          elseif cnter == 3 # float
              val_temp = parse(Float64,val_temp)
          elseif cnter == 4 # array floats
              val_temp = temp[1:end-2]
              val_temp = [parse(Float64,ss) for ss in val_temp]
          elseif cnter == 5 # optional CE bound params
              val_temp = parse(Float64,val_temp)
          end
          string_as_varname(var_temp,val_temp)
      end
    end
    close(f) # close doc
    cd("..")
end
