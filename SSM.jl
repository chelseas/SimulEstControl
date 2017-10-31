### Defining the state-space model for our problem

abstract type AbstractSSM end

### Nonlinear state-space model building off AbstractSSM
immutable NonLinearSSM <: AbstractSSM
    f
    Q::Array{Float64,2}
    h
    R::Array{Float64,2}
    nu
    nx
    ny
    states
    w
    v
    # Creating instance of NonLinearSSM
    function NonLinearSSM(f::Function,Q::Matrix,h::Function,R::Matrix,nu::Int,nx::Int,ny::Int,states::Int,w::MvNormal,v::MvNormal)
        new(f,Q,h,R,nu,nx,ny,states,w,v)
    end
end

### Defining state-space model matrices for 2D problem
function build2DSSM(deltaT::Float64,processNoise::Float64,measNoise::Float64)#w::MvNormal,v::MvNormal)
    #x = [vx, vy, ω, px, py, θ, m, μv, J, rx, ry]'
    #u = [Fx, Fy, Ta]

    ### Defining xdot = Ax + Bu
    # how should the functions for A, B, C, D be passed so it can take args from x and u?
    function f(x,u)
        firstOrder = deltaT*x[8]/x[7];
        secondOrder = (deltaT*x[8])^2/x[7];
        A = [1.0-firstOrder 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 1.0-firstOrder 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 1.0-firstOrder 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            deltaT-secondOrder 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 deltaT-secondOrder 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 deltaT-secondOrder 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
            0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0];

        T = (cos(x[6])*x[10]-sin(x[6])*x[11])*u[2] - (sin(x[6])*x[10]+cos(x[6])*x[11])*u[1]+u[3]

        B = [deltaT/x[7] 0.0 0.0;
            0.0 deltaT/x[7] 0.0;
            0.0 0.0 deltaT/x[9];
            deltaT^2/(2*x[7]) 0.0 0.0;
            0.0 deltaT^2/(2*x[7]) 0.0;
            0.0 0.0 deltaT^2/(2*x[9]);
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0];
        return A*x + B*[u[1];u[2];T];
    end

    ### Defining y = Cx+Du+r
    #go through and make sure this function is type stable -- check all the others really quickly
    #ask about the newH function in the filter method -- make sure filter is type stable
    #run the full script iteratively to double check that the full system is working
    function h(x,u) # removed the type constraints cuz EKF needs to take differential which can't have it
        #rx can't be converted to Float64

        rx = (cos(x[6])*x[10]-sin(x[6])*x[11])
        ry = (sin(x[6])*x[10]+cos(x[6])*x[11])

        term = -x[8]/x[7];

        C = [1.0 0.0 -ry 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 1.0 rx 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
            term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0];

        D = [0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            1/x[7] 0.0 0.0;
            0.0 1/x[7] 0.0;
            -ry/x[9] rx/x[9] 1/x[9]];

        # What is r?
        r = [0.0; 0.0; 0.0;
            rx; ry; 0.0;
            0.0; 0.0; 0.0];
        return C*x + D*u + r;
    end


    # Defining observability of system --> measuring all outputs
    velocitySensors = 1;
    positionSensors = 1;
    accelerometers = 1;
    #=
    ### Estimated output equations?
    function p(x,u)
        #rx can't be converted to Float64

        rx = (cos(x[6])*x[10]-sin(x[6])*x[11])
        ry = (sin(x[6])*x[10]+cos(x[6])*x[11])

        term = -x[8]/x[7];

        C = [1.0 0.0 -ry 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 1.0 rx 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0;
            term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
            0.0 0.0 term 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0];

        D = [0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            0.0 0.0 0.0;
            1/x[7] 0.0 0.0;
            0.0 1/x[7] 0.0;
            -ry/x[9] rx/x[9] 1/x[9]];

        r = [0.0; 0.0; 0.0;
            rx; ry; 0.0;
            0.0; 0.0; 0.0];

        return C*x + D*u + r;

    end
=#
    nx = 11;
    ny = 3*(velocitySensors+positionSensors+accelerometers);
    nu = 3;
    states = 6
    Q = diagm(processNoise*ones(nx))
    w = MvNormal(zeros(nx),Q) # process noise distribution
    v = MvNormal(zeros(ny),measNoise*eye(ny,ny)) #measurement noise distribution
    return NonLinearSSM(f,cov(w),h,cov(v),nu,nx,ny,states,w,v)
end

function buildDoubleIntSSM(deltaT::Float64,processNoise::Float64,measNoise::Float64)#w::MvNormal,v::MvNormal)
    f(x,u) = [1 0 0; deltaT 1 0; 0 0 1]*x + [deltaT/x[3]; deltaT^2/(x[3]^2);0]*u[1] # this is issue
    h(x,u) = [1 0 0; 0 1 0]*x
    nx = 3
    ny = 2
    nu = 1
    states = 2
    Q = diagm(processNoise*ones(nx))
    w = MvNormal(zeros(nx),Q) # process noise distribution
    v = MvNormal(zeros(ny),measNoise*eye(ny,ny)) #measurement noise distribution
    return NonLinearSSM(f,cov(w),h,cov(v),nu,nx,ny,states,w,v)
end
