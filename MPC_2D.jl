#Finite-horizon MPC controller
function MPCAction(x0::MvNormal, n::Int)
    xEst = mean(x0)
    unc = trace(cov(x0))
    x = mean(x0)

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

    # these were the values I used previously and I'm leaving just in case
    #=vg = 0.0#3
    pg = 300.0
    ga = 0.0#4
    Q  = [vg, vg, vg, pg, pg, pg]
    R = [ga, ga, 0.5*ga]=#
    Qp = diag(Qg)
    u = Variable(3, n - 1)
    xs = Variable(6, n)

    a = Float64

    # this should reflect the same cost function as the MCTS
    # need to add penalties to other states and actions
    problem = minimize(sum(abs.(xs[4,:])*Qp[4]) + sum(abs.(xs[5,:])*Qp[5]) + sum(abs.(xs[6,:])*Qp[6]))

    #sum(abs(xs[1,:])*Q[1]) + sum(abs(xs[2,:])*Q[2]) + sum(abs(xs[3,:])*Q[3]) + sum(abs(u[1,:])*R[1]) +sum(abs(u[2,:])*R[2]) + sum(abs(u[3,:])*R[3])
    problem.constraints += xs[1,1] == x[1] # init condition
    #problem.constraints += xs[1,n] == 0.0  # final cond
    problem.constraints += xs[2,1] == x[2] # init condition
    #problem.constraints += xs[2,n] == 0.0  # final cond
    problem.constraints += xs[3,1] == x[3] # init condition
    #problem.constraints += xs[3,n] == 0.0  # final cond
    problem.constraints += xs[4,1] == x[4] # init condition
    #problem.constraints += xs[4,n] == 0.0  # final cond
    problem.constraints += xs[5,1] == x[5] # init condition
    #problem.constraints += xs[5,n] == 0.0  # final cond
    problem.constraints += xs[6,1] == x[6] # init condition
    #problem.constraints += xs[6,n] == 0.0  # final cond

    for i in 1:n-1
        problem.constraints += xs[:, i+1] == A*xs[:, i] + B*u[:,i]
        problem.constraints += u[:,i] <= fRange
        problem.constraints += u[:,i] >= -fRange
    end

    # Solve the problem by calling solve!
    solve!(problem, SCSSolver(verbose=0))

     # Check the status of the problem
    problem.status # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimal value
    #a = problem.optval
    u_return = evaluate(u)

    return u_return[:,1]
end
