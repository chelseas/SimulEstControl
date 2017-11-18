#=
using PGFPlots

x = [1,2,3]
y = [2,4,1]
p = plot(x,y)
save("myfile.svg", p)
=#

using Plots
plotly()
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
losses_vec = [1.29889, 1.136, 1.14481, 0.843559, 0.976848, 1.09146, 0.767144, 1.14669, 1.27953, 0.905365, 0.95704, 0.940759, 0.687235, 0.893872, 0.543758, 0.445272, 0.447648, 0.907647, 0.875995, 0.624881, 0.463792, 0.765942, 0.29165, 0.430215, 0.166111, 0.341819, 0.21256, 0.154712, 0.206421, 0.179415, 0.257553, 0.107972, 0.146767, 0.184074, 0.217935, 0.216489, 0.0750545, 0.0656534, 0.0565964, 0.238641, 0.0781746, 0.0993684, 0.217716, 0.310835, 0.256997, 0.254831, 0.0766558, 0.084926, 0.107884, 0.108691, 0.122582, 0.217092, 0.0517251, 0.0740617, 0.0687145, 0.202962, 0.110812, 0.0410999, 0.111106, 0.0695679, 0.146494, 0.0814098, 0.1201, 0.126903, 0.0191794, 0.205839, 0.0734166, 0.0673355, 0.288253, 0.119778, 0.159527, 0.176905, 0.233972, 0.0275086, 0.193843, 0.0765488, 0.11253, 0.133522, 0.123583, 0.0867426, 0.291382, 0.0610016, 0.100401, 0.328839, 0.481978, 0.148418, 0.159774, 0.153781, 0.0587556, 0.105056, 0.129439, 0.0500423, 0.0692389, 0.280818, 0.150808, 0.125957, 0.0582144, 0.108673, 0.0992886, 0.200659]
plot(losses_vec)

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
