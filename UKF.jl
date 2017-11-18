# for use: xNew = ukf(ssm,obs[:,i],xNew,cov(w),cov(v),u[:,i])

### Correction term for trying to avoid Kalman numerical errors
function nearestSPD(A::Matrix{Float64})
    #Ahat and mineig are any
    Ahat::Array{Float64,2}
    #mineig::Array{Float64,1}
    n = size(A, 1)
    @assert(n == size(A, 2)) # ensure it is square

    I = eye(n)

    # symmetrize A into B
    B = (A+A')./2

    if isnan(B[1])
        B = 0.01*ones(ssm.nx,ssm.nx)
       @show B
    end
    # Compute the symmetric polar factor of B. Call it H.
    # Clearly H is itself SPD.
    U, σ, V = svd(B)
    H = V*diagm(σ)*V'

    # get Ahat in the above formula
    Ahat = (B+H)/2
    typeof(Ahat)
    Ahat
    # ensure symmetry
    Ahat = (Ahat + Ahat')/2;

    # test that Ahat is in fact PD. if it is not so, then tweak it just a bit.
    worked = false
    iteration_count = 0

    # tweak the matrix so that it avoids the numerical instability
    while !worked && iteration_count < 100
        iteration_count += 1

        try
            chol(Ahat)
            worked = true
        catch
            ## ERR: matrix wasn't able to be fixed!
            if iteration_count == 10
              @show msg = "failed"
            end
        end

        if !worked
            # Ahat failed the chol test. It must have been just a hair off,
            # due to floating point trash, so it is simplest now just to
            # tweak by adding a tiny multiple of an identity matrix.

            min_eig = minimum(eigvals(Ahat))
            Ahat = Ahat + (-min_eig*iteration_count.^2 + eps(Float64))*I
        end
    end

    Ahat
end

### Checking a matrix during Kalman filtering
function robust_chol(A::Array{Float64,2})
    C = try
        chol(A)
    catch
        chol(nearestSPD(A))
    end
    return C
end

# UKF implementation
function ukf(m::NonLinearSSM, # NL SSM
            z::Vector{Float64}, # obs
            x0::MvNormal,
            Q::Array{Float64,2}, # process noise
            R:: Array{Float64,2}, # measurement noise
            u::Array=zeros(m.nu,size(z,2)))

    # Aliases (notations from UKF papers)
    α = 0.9#1e-3
    κ = 0. # also try 1 here --> not sure why?
    β = 2.
    x = mean(x0)
    P = cov(x0)
    n = length(x)
    l = length(z)

    # Calculate Sigma Points
    λ = α^2*(n+κ)-n
    γ = sqrt(n+λ)
    X = sigma(x,P,γ) # sigma points
    # Update using unscented transform
    # weights
    Wm = [λ/(n+λ) 0.5/(n+λ)+zeros(1,2*n)]
    Wc = [λ/(n+λ)+(1-α^2+β) 0.5/(n+λ)+zeros(1,2*n)]
    x_,P_,Ax,z_,Pz_,Az = ut(m,X,Q,R,u,Wm,Wc,n,l)   # unscented transform of process
    Pxy = Ax*cat([1,2],Wc...)*Az'  # Cross covariance
    K = Pxy*inv(Pz_)   # Kalman gain

    # state update
    a = x_ + K*(z-z_) # updated estimate
    x_new = a[:,1] # new estimate mean

    cov_new = nearestSPD(P_-K*Pxy') # new estimate cov
    return MvNormal(x_new,cov_new)
end

# Selecting the sigma points
function sigma(x::Vector{Float64},P::Array{Float64,2},γ::Float64) # x = reference pt, P = covariance, c = coeff
    n = length(x)
    # P_ = nearestSPD(P)
    S = chol(P) # replace P with P_?
    Y = zeros(n,n)
    for i=1:n
        Y[:,i] = x
    end
    X = [x Y+γ*S Y-γ*S]
    return X # sigma points
end

# Unscented transformation
# function, additive cov, sigma pts, mean/cov weights, num outputs of f
function ut(m::NonLinearSSM,X,Q::Array{Float64,2},R::Array{Float64,2},
            u::Vector{Float64},Wm::Array{Float64,2},
            Wc::Array{Float64,2},n::Int64,l::Int64)
    L = size(X,2) # number of sigma points
    X_ = zeros(n,L) # transformed sigma pts
    x_ = zeros(n,1) # transformed sigma pts mean
    Y = zeros(n,L) # can this be merged w for loop above?
    Xo_ = zeros(l,L) # transformed sigma pts
    xo_ = zeros(l,1) # transformed sigma pts mean
    Yo = zeros(l,L) # can this be merged w for loop above?

    # ut for states
    for i=1:L
        X_[:,i] = m.f(X[:,i],u)
        x_ = x_ + Wm[i]*X_[:,i] #
        Xo_[:,i] = m.h(X[:,i],u)
        xo_ = xo_ + Wm[i]*Xo_[:,i] #
    end

    for i=1:L # for all sigma points, store
        Y[:,i] = x_
        Yo[:,i] = xo_
    end
        A = (X_-Y)
        P_  = A*cat([1,2],Wc...)*A'+Q
        Ao = (Xo_-Yo)
        Po_  = Ao*cat([1,2],Wc...)*Ao'+R

        return x_,P_,A,xo_,Po_,Ao
end
