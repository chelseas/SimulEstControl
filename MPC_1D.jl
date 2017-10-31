#Finite-horizon MPC controller
function MPCAction(x0::MvNormal,n::Int)
    xEst = mean(x0)
    m = xEst[3]
    pos = [xEst[1], xEst[2]]
    unc = trace(cov(x0))   # uncertainty in states, can I set to zero?
    #goal = [0.0; 0.0]
    A = [1 0; deltaT 1];
    B = [(deltaT/m);(deltaT^2/(2*m))]

    # gains
    #ga = .008#0.01  -10 w/ fDist at 50; #gain for force cost  #force^2 maxes at 25000
    #gu = 10#-50;5 #gain for uncertainty #trace^2 maxes at 9
    #gp = 10#3   -20; #gain for position    #pos^2 starts 400 goes to 0
    #gv = 0.3

    #ga = 0.04#0.4  -10 w/ fDist at 50; #gain for force cost  #force^2 maxes at 25000
    #gp = 300.0#3   -20; #gain for position    #pos^2 starts 400 goes to 0
    #gv = 0.0
    #ga = 1.0#0.01  -10 w/ fDist at 50; #gain for force cost  #force^2 maxes at 25000
    #gp = 10.0#3   -20; #gain for position    #pos^2 starts 400 goes to 0
    #gv = 3.0
    (gv, gp) = diag(Qg)
    #col vector variable size nx1
    u = Variable(1, n - 1)
    x = Variable(2, n)

    a = Float64

    #optimization problem
    problem = minimize(sum(abs.(x[1,:]))*gv + sum(abs.(x[2,:]))*gp + sum(abs.(u[1,:]))*Rg)# + unc^2*gu)#  + sumsquares(trace(diagm(unc)))*gUnc) #add uncertainty  + sumsquares(trace)*gUnc


    #motion constraints
    #problem.constraints += x[3,1] == m
        #init/final constraints
    problem.constraints += x[2,1] == pos[1] # init condition
    problem.constraints += x[2,n] == 0.0  # final cond
    problem.constraints += x[1,1] == pos[2]
    problem.constraints += x[1,n] == 0.0  # final cond on veloc
    #can either use positions as parameters or as variables with fixed constraints for the propogation
    for i in 1:n-1
        problem.constraints += x[:, i+1] == A*x[:, i] + B*u[i] #ssm.f(x[:,i], u[i]) #+ rand(w)#x[2,i] + deltaT*x[1,i]
        #accel = u[i]/m
        #problem.constraints += x[1, i+1] == #x[1, i]+deltaT*accel
    end

    #force limits
    problem.constraints += u <= fRange
    problem.constraints += u >= -fRange



    # Solve the problem by calling solve!
    # solve!(problem, ECOSSolver())
    solve!(problem, SCSSolver(verbose=0))

     # Check the status of the problem
    problem.status # :Optimal, :Infeasible, :Unbounded etc.

    # Get the optimal value
    #a = problem.optval
    u_return = evaluate(u)

    return u_return[1,1]
end
