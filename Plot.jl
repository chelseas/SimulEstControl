#=
using PGFPlots

x = [1,2,3]
y = [2,4,1]
p = plot(x,y)
save("myfile.svg", p)
=#

using Plots
plotly()
# pyplot()
#plot(rand(3,5)) # plots 5 lines of 3 data points

# assuming we want to plot the individual rows of data
function plot_data(data::Matrix, names::Vector, title::String, xlabel::String, ylabel::String)
#num_matrices = length(data) # m rows, n columns of data
#@show names
  plot(data', linewidth=2, title=title, xlab=xlabel, ylab=ylabel, label=names[1:end])
end





# assuming we want to plot the individual rows of data
function plot_tuple(data::Tuple)#, names::Vector, title::String, xlabel::String, ylabel::String)
  num_matrices = length(data) # m rows, n columns of data
  (a,b,c,d) = data#@show names
  #plot(data', linewidth=2, title=title, xlab=xlabel, ylab=ylabel, label=names[1:end])
  p1 = plot(b')
  p2 = plot(b')
  a = [p1 p2]
  plot(a,layout=(2,1))
end


# example test plotting 3 vectors of length 100
b = rand(3,100);
title = "Test Title"
xlabel = "X axis"
ylabel = "Y axis"
legend = ["1","2","3"]
d = (b,b,b,b)
if 1 ==1
  plot_data(b,legend,title,xlabel,ylabel)
end
#plot_data(d)
