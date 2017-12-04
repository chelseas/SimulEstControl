# Finite-horizon MPC controller
function MPCAction(x0::MvNormal,n::Int)
    xEst = mean(x0)
    m = xEst[3]
    pos = [xEst[1], xEst[2]]

    A = [1 0; deltaT 1];
    B = [(deltaT/m);(deltaT^2/(2*m))]

    (gv, gp) = diag(Qg)
    #col vector variable size nx1
    u = Variable(1, n - 1)
    x = Variable(2, n)

    a = Float64

    #optimization problem
    problem = minimize(sum(abs.(x[1,:]))*gv + sum(abs.(x[2,:]))*gp + sum(abs.(u[1,:]))*Rg)# + unc^2*gu)#  + sumsquares(trace(diagm(unc)))*gUnc) #add uncertainty  + sumsquares(trace)*gUnc

    problem.constraints += x[2,1] == pos[1] # init condition
    problem.constraints += x[2,n] == 0.0  # final cond
    problem.constraints += x[1,1] == pos[2]
    problem.constraints += x[1,n] == 0.0  # final cond on veloc
    for i in 1:n-1
        problem.constraints += x[:, i+1] == A*x[:, i] + B*u[i]
    end

    #force limits
    problem.constraints += u <= fRange
    problem.constraints += u >= -fRange

    # Solve the problem by calling solve!
    solve!(problem, SCSSolver(verbose=0))

     # Check the status of the problem
    problem.status # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimal action
    u_return = evaluate(u)
    Convex.clearmemory() # stops it from creating new vars each time
    return u_return[1,1]
end
