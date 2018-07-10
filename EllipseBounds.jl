# input samples and arrays of bounds same length as first dimension of samples
# this clips samples to those bounds
function bound_estimates(samples::Array{Float64,2}, lb::Array{Float64,1}, ub::Array{Float64,1})
    p,n_out = size(samples) # dims by number of output sample points
    # make sure p and elements in lb/ub are same
    for i in 1:p
        lb_lim = lb[i]
        ub_lim = ub[i]

        lb_idx = samples[i,:] .> lb_lim # indices (list) of points where lb_lim should be enforced
        #samples[i,lb_idx] = lb_lim
        samples = samples[:,lb_idx]
        #@show i,size(samples)
        ub_idx = samples[i,:] .< ub_lim
        #samples[i,ub_idx] = ub_lim
        samples = samples[:,ub_idx]
    end
    try
        samples = samples[:,1:n_out_w]
    catch
    end
    return samples
end

# Pass in a MvNormal, number of samples to compute ellipse, number of sapmles to output of ellipsoid, F statistic
# F-distrib calculated value based on p, n-p, alpha, Stat Trek
function ellipsoid_bounds(A::MvNormal,n::Int64,n_out::Int64,F::Float64) # finds ellispoidal bounds of MvNormal passed into it
    #the F is now just the multipler for the eigenvalues, n value no longer used
    mu = mean(A) # mean of distribution
    Σ = cov(A)
    p = length(mu) # number of degrees of freedom
    #s = rand(A,n) # samples of the desired distribution A, p x n
    #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
    #x_bar = mean(sample_mv)
    #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary

    # Sampling from a p-dimensional sphere will be projected onto confindence ellipse
    sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
    sphere_pts = rand(sphere,n_out) # p x n_out of rand values
    lambdas = sqrt.(sum(sphere_pts.^2,1)) # computing normalizing factor to put sphere_pts on sphere surface
    sphere_unit_pts = sphere_pts./lambdas # points on the unit sphere

    U,S,V = svd(Σ)#Σ_hat_mean) # SVD of sampled ellipse
    #S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse
    S2 = S*F

    ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*sphere_unit_pts) # proj pnts to confidence ellipse and add offset mu

  if clip_bounds
      ellipse_pts = bound_estimates(ellipse_pts, lb_clip_lim, ub_clip_lim)
  end

  return ellipse_pts # returns p x n_out array
end

function ellipsoid_bounds_noclip(A::MvNormal,n::Int64,n_out::Int64,F::Float64) # finds ellispoidal bounds of MvNormal passed into it
    #the F is now just the multipler for the eigenvalues, n value no longer used
    mu = mean(A) # mean of distribution
    Σ = cov(A)
    p = length(mu) # number of degrees of freedom
    #s = rand(A,n) # samples of the desired distribution A, p x n
    #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
    #x_bar = mean(sample_mv)
    #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary

    # Sampling from a p-dimensional sphere will be projected onto confindence ellipse
    sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
    sphere_pts = rand(sphere,n_out) # p x n_out of rand values
    lambdas = sqrt.(sum(sphere_pts.^2,1)) # computing normalizing factor to put sphere_pts on sphere surface
    sphere_unit_pts = sphere_pts./lambdas # points on the unit sphere

    U,S,V = svd(Σ)#Σ_hat_mean) # SVD of sampled ellipse
    #S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse
    S2 = S*F

    ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*sphere_unit_pts) # proj pnts to confidence ellipse and add offset mu
    return ellipse_pts # returns p x n_out array
end

# setting a given covariance for the ellipsoid below to make easier to check
function ellipsoid_inner(A::MvNormal,n::Int64,n_out::Int64,F::Float64)

      #the F is now just the multipler for the eigenvalues, n value no longer used
      mu = mean(A) # mean of distribution
      Σ = cov(A)
      p = length(mu) # number of degrees of freedom
      #s = rand(A,n) # samples of the desired distribution A, p x n
      #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
      #x_bar = mean(sample_mv)
      #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary

        # sampling points uniformly inside unit circle
        sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
        Y = rand(sphere,n_out)#*2-1 #
        U = rand(n_out) # 0 to 1
        r = U.^(1/p) # rad prop to d_th root of U
        ssrow = sqrt.(sum(Y.^2,1))'
        scaled = Y./ssrow'
        X = r'.*scaled

      U,S,V = svd(Σ)#Σ_hat_mean) # SVD of sampled ellipse
      #S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse
      S2 = S*F
      ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*X) # proj pnts to confidence ellipse and add offset mu

  if clip_bounds
      ellipse_pts = bound_estimates(ellipse_pts, lb_clip_lim, ub_clip_lim)
  end

  return ellipse_pts
end

function ellipsoid_bounds_plus_inner(A::MvNormal,n::Int64,n_out::Int64,F::Float64) # finds ellispoidal bounds of MvNormal passed into it
    mu = mean(A) # mean of distribution
    Σ = cov(A)
    p = length(mu) # number of degrees of freedom
    #s = rand(A,n) # samples of the desired distribution A, p x n
    #sample_mv = fit(typeof(A),s) # MLE mean of the sample estimate (not unbiased)
    #x_bar = mean(sample_mv)
    #Σ_hat_mean = n*cov(sample_mv) # the n is for the sampled MvNormal region --> necessary

    # Sampling from a p-dimensional sphere will be projected onto confindence ellipse
    sphere = MvNormal(zeros(p),eye(p,p)) # zero centered with rad 1
    sphere_pts = rand(sphere,n_out) # p x n_out of rand values
    lambdas = sqrt.(sum(sphere_pts.^2,1)) # computing normalizing factor to put sphere_pts on sphere surface
    sphere_unit_pts = sphere_pts./lambdas # points on the unit sphere

    U,S,V = svd(Σ)#Σ_hat_mean) # SVD of sampled ellipse
    #S2 = sqrt.(abs.(S))*sqrt((p*(n-1)*F)/(n*(n-p))) # new eigenvalues of confidence ellipse
    S2 = S*F

      Y = rand(sphere,n_out)#*2-1 #
      Us = rand(n_out) # 0 to 1
      r = Us.^(1/p) # rad prop to d_th root of U
      ssrow = sqrt.(sum(Y.^2,1))'
      scaled = Y./ssrow'
      X = r'.*scaled

    ellipse_pts = broadcast(+,mu,U*diagm(S2)*V'*sphere_unit_pts) # proj pnts to confidence ellipse and add offset mu
    ellipse_inner_pts = broadcast(+,mu,U*diagm(S2)*V'*X) # proj pnts to confidence ellipse and add offset mu

  if clip_bounds
      ellipse_pts = bound_estimates(ellipse_pts, lb_clip_lim, ub_clip_lim)
      ellipse_inner_pts = bound_estimates(ellipse_inner_pts, lb_clip_lim, ub_clip_lim)
  end

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
  # F computed with n = 100, p = 11, alpha = 0.05, F(p,n-p,alpha) = 1.9
  n = 100 # default samples for computing the est ellipsoid_bounds
  F = 1.9 #2.34 # computed offline based on F-stat for alpha=0.05,n=20,p=5
  n_out = 1000 # number of samples of the ellipsoid bound initially taken before cropping
  #@show est
  est_bound = ellipsoid_inner(est,n,n_out,F) # compute points on est_ellipse, size dim of sys x n_out
  #est_bound[:,1]
  dim, n_out_corrected = size(est_bound)
  if n_out_corrected == 0
      state_bound = desired_bounds
  else
      state_bound = 0 # set initially to 0
  end
  for i = 1:n_out_corrected
    if (length(state)==1) && state[1] == -100.0
      sample_state = est_bound[:,i] # form combined "state" from state and est samples
    else
      sample_state = [state; est_bound[:,i]] # form combined "state" from state and est samples
    end
    sample_next_state = ssm.f(sample_state,u) # propagating the sample_state and given action forward
    sample_state_bound = norm(sample_next_state[1:6]) # taking norm
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
