### Defining the state-space model for our problem

abstract type AbstractSSM end

### Nonlinear state-space model building off AbstractSSM
immutable NonLinearSSM <: AbstractSSM
    f
    h
    nu
    nx
    ny
    states

    function NonLinearSSM(f::Function,h::Function,nu::Int,nx::Int,ny::Int,states::Int)
        new(f,h,nu,nx,ny,states)
    end
end

### Defining state-space model matrices for 2D problem
function build2DSSM(deltaT::Float64)
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
        return C*x + D*u + r;  # adding some nonlinearity due to inertia offsets
    end

    # Defining observability of system --> measuring all outputs
    velocitySensors = 1;
    positionSensors = 1;
    accelerometers = 1;

    nx = 11;
    ny = 3*(velocitySensors+positionSensors+accelerometers);
    nu = 3;
    states = 6
    return NonLinearSSM(f,h,nu,nx,ny,states)
end

function buildDoubleIntSSM(deltaT::Float64)
    f(x,u) = [1 0 0; deltaT 1 0; 0 0 1]*x + [deltaT/x[3]; deltaT^2/(x[3]^2);0]*u[1] # this is issue
    h(x,u) = [1 0 0; 0 1 0]*x
    nx = 3
    ny = 2
    nu = 1
    states = 2

    return NonLinearSSM(f,h,nu,nx,ny,states)
end

function buildCarSSM(deltaT::Float64)
    #X = [x, y, theta, v, mu]' (x coordinate, y coordinate, steering angle, speed, coefficient of friction)
    #u = [omega, a]' (steering input, acceleration)

    #xdot = f(X, u)
    #Xdot = [xdot, ydot, thetadot, vdot, mudot]'
    #f(X, u) = [v cos(theta), v sin(theta), v omega, a - mu v, 0]'
    #.
    #.
    #.
    #X(t + 1) = fd(X(t), u(t))
    #X(t + 1) = [x(t + 1), y(t + 1), theta(t + 1), v(t + 1), mu(t + 1)]'
    #fd(X(t), u(t)) = [x(t) + dt(v(t) cos(theta(t))), y(t) + dt(v(t) sin(theta(t))), theta(t) + dt(v(t) omega(t)), v(t) + dt(a(t) - mu(t) v(t)), mu(t)]

    function f(x,u)
      sp = [x[1] + deltaT*(x[4]*cos(x[3]));
              x[2] + deltaT*(x[4]*sin(x[3]));
              x[3] + deltaT*(x[4]*u[1]);
              x[4] + deltaT*(u[2] - x[5]*x[4]);
              x[5]];
      return sp
    end

    #Yt = ht(X(t), u(t))
    #ht(X(t), u(t)) = [x(t), y(t), theta(t), v(t)]
    function h(x,u)
        return x[1:(end - 1)]
    end

    # Defining observability of system
    nx = 5;
    ny = 4;
    nu = 2;
    states = 4;
    return NonLinearSSM(f,h,nu,nx,ny,states)
end
