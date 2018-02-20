#using Distributions, Convex, ECOS, SCS

#Finite-horizon MPC controller
function MPCAction(x0::MvNormal, n::Int, k::Int) # k is the number of steps where the problem is tried
    unc = trace(cov(x0))
    x = mean(x0)
    # Check this jacobian with forward diff to make sure these are linearized
    # Dynamics for 2D problem
    firstOrder = deltaT*x[8]/x[7];
    secondOrder = (deltaT*x[8])^2/x[7];

    A = [1-firstOrder 0 0 0 0 0;
        0 1-firstOrder 0 0 0 0;
        0 0 1-firstOrder 0 0 0;
        deltaT-secondOrder 0 0 1 0 0;
        0 deltaT-secondOrder 0 0 1 0;
        0 0 deltaT-secondOrder 0 0 1;];

    B = [deltaT/x[7] 0 0;
        0 deltaT/x[7] 0;
        0 0 deltaT/x[9];
        deltaT^2/(2*x[7]) 0 0;
        0 deltaT^2/(2*x[7]) 0;
        0 0 deltaT^2/(2*x[9]);];

    # can speed up slight ~20% if I put the variables or objects outside of the for loop
    Qp = diag(Qg)
    Rp = diag(Rg)
    u = Convex.Variable(3, n - 1)
    xs = Convex.Variable(6, n)
    w = MvNormal(zeros(ssm.states),Q[1:ssm.states,1:ssm.states]) # defining process noise
    w_sample = rand(w,n-1) # taking n samples from process noise for propagating


    problem = minimize(sum(abs.(u[1,:])*Rp[1]) + sum(abs.(u[2,:])*Rp[2]) + sum(abs.(u[3,:])*Rp[3]) + sum(abs.(xs[1,:])*Qp[1]) + sum(abs.(xs[2,:])*Qp[2]) + sum(abs.(xs[3,:])*Qp[3]) + sum(abs.(xs[4,:])*Qp[4]) + sum(abs.(xs[5,:])*Qp[5]) + sum(abs.(xs[6,:])*Qp[6]))

    problem.constraints += xs[1,1] == x[1] # init condition
    problem.constraints += xs[2,1] == x[2] # init condition
    problem.constraints += xs[3,1] == x[3] # init condition
    problem.constraints += xs[4,1] == x[4] # init condition
    problem.constraints += xs[5,1] == x[5] # init condition
    problem.constraints += xs[6,1] == x[6] # init condition


    for i in 2:n-k#if reward_type == "region"
        problem.constraints += xs[4,i] <= region_ub[1]  # final cond
        problem.constraints += xs[4,i] >= region_lb[1]  # final cond
        problem.constraints += xs[5,i] <= region_ub[2]  # final cond
        problem.constraints += xs[5,i] >= region_lb[2]  # final cond
        problem.constraints += xs[6,i] <= region_ub[3]  # final cond
        problem.constraints += xs[6,i] >= region_lb[3]  # final cond
    end

    for i in 1:n-1
        problem.constraints += xs[:, i+1] == A*xs[:, i] + B*u[:,i] + w_sample[:,i]# system with process noise
        problem.constraints += u[:,i] <= fRange
        problem.constraints += u[:,i] >= -fRange
    end

    # Solve the problem by calling solve!
    solve!(problem, ECOSSolver(verbose=0))

     # Check the status of the problem
    #@show problem.status # :Optimal, :Infeasible, :Unbounded etc.
    if problem.status == :Infeasible
        u_error = [-10.0,-10.0,-10.0]
        return u_error
    end
    # Get the optimal value
    u_return = Convex.evaluate(u)
    Convex.clearmemory() # stops it from creating new vars each time
    return u_return[:,1]
end

function MPCActionConstrained(x0::MvNormal, n::Int, k::Int)
    u_error = [-10.0,-10.0,-10.0]
    for i = 0:k
        #@show i
        u_temp = MPCAction(x0, n, i)
        if u_temp != u_error # real action should be returned
            return u_temp
        end
    end
    return [0.0,0.0,0.0] # if the problem is infeasible for all setups return all 0's
end

# Variables for testing MPC_Constrained_2D
#=
size = 11
x0 = MvNormal(ones(11),eye(11))
n = 50
k = 50
deltaT = 0.1
fRange = 5.0 # bounds on controls allowed within +- fRange
state_init = 1.0 # gain for the initial state
fDist_disc = 1000 # discrete points in fDist force linspace
reward_type = "none"
state_bound_value = 2.0 #0.5
th_bound_value = 3.14159/2
region_lb = [-state_bound_value, -state_bound_value, -th_bound_value] # values for x,y,theta lower bound for being in the reward area
region_ub = [state_bound_value, state_bound_value, th_bound_value]
rew_in_region = 0.0
rew_out_region = -1.0
# Reward shaping
Qg = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # for use in POMDP reward funct
Qr = 1.0*diagm([0.3, 0.3, 0.3, 50.0, 50.0, 50.0]) # for computing reward below just for measured states
Rg = 0.25*diagm([10.0, 10.0, 10.0])
a = MPCActionConstrained(x0, n, k)
@show a
=#
