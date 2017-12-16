# Pass in a MvNormal, number of samples to compute ellipse, number of sapmles to output of ellipsoid, F statistic
# F-distrib calculated value based on p, n-p, alpha, Stat Trek
function ellipsoid_bounds(A::MvNormal,n::Int64,n_out::Int64,F::Float64) # finds ellispoidal bounds of MvNormal passed into it
  mu = mean(A) # mean of distribution
  p = length(mu) # number of degrees of freedom
  s = rand(A,n) # samples of the desired distribution A, p x n
  #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
  #x_bar = mean(sample_mv)
  #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary
  x_bar = mean(s,2)
  Σ_hat_mean = n*(1.0/(n))*(s - x_bar*ones(1,n))*(s - x_bar*ones(1,n))'

  # Sampling from a p-dimensional sphere will be projected onto confindence ellipse
  sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
  sphere_pts = rand(sphere,n_out) # p x n_out of rand values
  lambdas = sqrt.(sum(sphere_pts.^2,1)) # computing normalizing factor to put sphere_pts on sphere surface
  sphere_unit_pts = sphere_pts./lambdas # points on the unit sphere

  U,S,V = svd(Σ_hat_mean) # SVD of sampled ellipse
  S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse

  ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*sphere_unit_pts) # proj pnts to confidence ellipse and add offset mu
  return ellipse_pts # returns p x n_out array
end

# setting a given covariance for the ellipsoid below to make easier to check
function ellipsoid_inner(A::MvNormal,n::Int64,n_out::Int64,F::Float64)

  mu = mean(A) # mean of distribution
  p = length(mu) # number of degrees of freedom
  s = rand(A,n) # samples of the desired distribution A, p x n
  #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
  #x_bar = mean(sample_mv)
  #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary
  x_bar = mean(s,2)
  Σ_hat_mean = n*(1.0/(n))*(s - x_bar*ones(1,n))*(s - x_bar*ones(1,n))'

    # sampling points uniformly inside unit circle
    sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
    Y = rand(sphere,n_out)#*2-1 #
    U = rand(n_out) # 0 to 1
    r = U.^(1/p) # rad prop to d_th root of U
    ssrow = sqrt.(sum(Y.^2,1))'
    scaled = Y./ssrow'
    X = r'.*scaled

  U,S,V = svd(Σ_hat_mean) # SVD of sampled ellipse
  S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse
  ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*X) # proj pnts to confidence ellipse and add offset mu

  return ellipse_pts
end

function ellipsoid_bounds_plus_inner(A::MvNormal,n::Int64,n_out::Int64,F::Float64) # finds ellispoidal bounds of MvNormal passed into it
  mu = mean(A) # mean of distribution
  p = length(mu) # number of degrees of freedom
  s = rand(A,n) # samples of the desired distribution A, p x n
  #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
  #x_bar = mean(sample_mv)
  #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary
  x_bar = mean(s,2)
  Σ_hat_mean = n*(1.0/(n))*(s - x_bar*ones(1,n))*(s - x_bar*ones(1,n))'

  # Sampling from a p-dimensional sphere will be projected onto confindence ellipse
  sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
  sphere_pts = rand(sphere,n_out) # p x n_out of rand values
  lambdas = sqrt.(sum(sphere_pts.^2,1)) # computing normalizing factor to put sphere_pts on sphere surface
  sphere_unit_pts = sphere_pts./lambdas # points on the unit sphere

  U,S,V = svd(Σ_hat_mean) # SVD of sampled ellipse
  S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse

    Y = rand(sphere,n_out)#*2-1 #
    Us = rand(n_out) # 0 to 1
    r = Us.^(1/p) # rad prop to d_th root of U
    ssrow = sqrt.(sum(Y.^2,1))'
    scaled = Y./ssrow'
    X = r'.*scaled

  ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*sphere_unit_pts) # proj pnts to confidence ellipse and add offset mu
  ellipse_inner_pts = broadcast(+,mu,U*diagm(S2)*V'*X) # proj pnts to confidence ellipse and add offset mu
  return (ellipse_pts, ellipse_inner_pts) # returns p x n_out array
end

### Test Below, uncomment to run
#=
#using Distributions
# check which sigma I should be using
# n=10,p=2,alpha=0.05 ==> F = 4.46
# n = 30,p=11,alpha=0.05 ==> F = 2.34
# n = 51,P=11,alpha=0.05 ==> F = 2.04
# n = 20,p=5,alpha=0.05 ==> F = 2.9
F = 2.97 # F-distrib calculated value based on p, n-p, alpha, Stat Trek
p = 3 # number of dims for testing
n = 30 # num of samples taken
alpha = 0.05 # prob = 1 - alpha is confidence percentage
mu = zeros(p) # mean of distribution
d = MvNormal(mu,eye(p,p)) # distribution that we are trying to find the ellipse of and sample points from
ellipse_pts = ellipsoid_bounds(d,n,100,F)
using Plots
plotly()
#scatter(sphere_unit_pts[1,:],sphere_unit_pts[2,:]) # should be on unit circle/sphere
scatter(ellipse_pts[1,:],ellipse_pts[2,:],ellipse_pts[3,:]) # should be on unit circle/sphere scaled by lengths
=#

# compute the total bounded norm of the state, worst est params, action, and precomputed noise bound
# should the state be the belief and sample from the boundary of that as well? Here we're assuming the state fully measured
# should be able to pass in state as null array and est as the belief and it will work?
function overall_bounds(state::Array{Float64,1},est::MvNormal,u::Array{Float64,1},w_bound::Float64)
  n = 100 # default samples for computing the est ellipsoid_bounds
  F = 1.93 #2.34 # computed offline based on F-stat for alpha=0.05,n=20,p=5
  n_out = 100 # number of samples of the ellipsoid bound to return here to iter through
  est_bound = ellipsoid_bounds(est,n,n_out,F) # compute points on est_ellipse, size dim of sys x n_out
  est_bound[:,1]
  state_bound = 0 # set initially to 0
  for i = 1:n_out
    if (length(state)==1) && state[1] == -100.0
      sample_state = est_bound[:,i] # form combined "state" from state and est samples
    else
      sample_state = [state; est_bound[:,i]] # form combined "state" from state and est samples
    end
    sample_next_state = ssm.f(sample_state,u) # propagating the sample_state and given action forward
    sample_state_bound = norm(sample_next_state) # taking norm
    if sample_state_bound > state_bound
      state_bound = sample_state_bound # keep largest bound from samples
    end
  end
  total_bound = state_bound + w_bound # summing overall state and w bounds
  return total_bound
end

# need to load ssm so have to run a test in the main loop
#=
state = ones(6) # sample state
est = MvNormal(ones(5),eye(5,5)) # est Params
u = 3.0*ones(3) # sample action
w_bound = 0.1*11 # sample noise
@show bounds = overall_bounds(state,est,u,w_bound)
=#
