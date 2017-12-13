#Finite-horizon MPC controller
function MPCAction(x0::MvNormal, n::Int)
    xEst = mean(x0)
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

    problem = minimize(sum(abs.(u[1,:])*Rp[1]) + sum(abs.(u[2,:])*Rp[2]) + sum(abs.(u[3,:])*Rp[3]) + sum(abs.(xs[1,:])*Qp[1]) + sum(abs.(xs[2,:])*Qp[2]) + sum(abs.(xs[3,:])*Qp[3]) + sum(abs.(xs[4,:])*Qp[4]) + sum(abs.(xs[5,:])*Qp[5]) + sum(abs.(xs[6,:])*Qp[6]))

    problem.constraints += xs[1,1] == x[1] # init condition
    #problem.constraints += xs[1,n] == 0.0  # final cond
    problem.constraints += xs[2,1] == x[2] # init condition
    problem.constraints += xs[3,1] == x[3] # init condition
    problem.constraints += xs[4,1] == x[4] # init condition
    problem.constraints += xs[5,1] == x[5] # init condition
    problem.constraints += xs[6,1] == x[6] # init condition

    for i in 1:n-1
        problem.constraints += xs[:, i+1] == A*xs[:, i] + B*u[:,i]
        problem.constraints += u[:,i] <= fRange
        problem.constraints += u[:,i] >= -fRange
    end

    # Solve the problem by calling solve!
    solve!(problem, ECOSSolver(verbose=0))

     # Check the status of the problem
    problem.status # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimal value
    u_return = Convex.evaluate(u)
    Convex.clearmemory() # stops it from creating new vars each time
    return u_return[:,1]
end
