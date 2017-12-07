### control limiting for the car problem
control_limit = 1.05*fRange
control_lim = fRange

function control_check(u::Array{Float64,1}, x::Array{Float64,1}, flag::Bool)
  for i=1:length(u)
    if(u[i] > control_limit[i])
      u[i] = control_lim[i]
    elseif(u[i] < -control_limit[i])
      u[i] = -control_lim[i]
    end
  end
  return u
end

function Car_reward(x::Array{Float64, 1}, u::Array{Float64, 1}, idx::Int)
  # Find error in deviation from path
  Dist2State = sqrt.((x[1] - PathX).^2 + (x[2] - PathY).^2);
  DistErr = minimum(Dist2State);
  # Find error in distance from track point
  TrackErr = sqrt.((x[1] - PathX[idx])^2 + (x[2] - PathY[idx])^2)
  # Find error in orientation angle
  AngErr = abs.(atan((PathY[idx] - x[2])/(PathX[idx] - x[1]) - x[3]));
  # Find error in speed
  SpeedErr = abs.(SpeedLimit - x[4]);
  # reward nearness to path and speed limit
  r = -Kdist*DistErr - Kspeed*SpeedErr - Ktrack*TrackErr - Kang*AngErr;
  return r
end

