### Correction term for trying to avoid Kalman numerical errors
function nearestSPD(A::Matrix{Float64})
    #Ahat and mineig are any
    # Ahat::Array{Float64,2}
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

### Predict step in algo
# takes in a state space model, previous guess, and action, to predict new state distribution
function predict(m::NonLinearSSM,x_prev::MvNormal,Q::Array{Float64,2},u::Array=zeros(m.nu,1))
    mu0 = mean(x_prev)'';
    P0 = cov(x_prev)'';
    newF(x) = ssm.f(x, u) # newF(x::Matrix) = ssm.f(x,u)
    results = zeros(ssm.nx,ssm.nx)
    F = ForwardDiff.jacobian!(results,newF,mu0) #getting any on this var
    #@show F = ForwardDiff.jacobian(newF,mu0)

    mu_pred = ssm.f(mu0,u) # mu_pred::Array{Float64,2} = ssm.f(mu0,u) # getting any on this var
    #@show mu_pred = ssm.f(mu0,u)
    #conversion from 1 type always same to another not that bad -- conversion from any to 1 type
    P_pred::Array{Float64,2} = convert(Array{Float64,2},Hermitian(F*P0*F' + Q)) #getting any on this var
    P_pred = nearestSPD(P_pred)
    return MvNormal(mu_pred,P_pred) # MvNormal(mu_pred[:],P_pred)
end

### Extended Kalman Filter update
#Given a model, observation, previous distribution, and action, updates the Kalman filter
#to generate a new distribution over the state space
function filter(m::NonLinearSSM,obs::Array{Float64,1},x0::MvNormal,Q::Array{Float64,2},R::Array{Float64,2},u::Array=zeros(m.nu,size(obs,2)))
    x_new = x0
    for i = 1:size(obs,2)
        x_pred = predict(m,x_new,Q,u[:,i]'')
        #@show "here"
        x_pred_mean = mean(x_pred)
        #@show u
        newobs = m.h(x_pred_mean,u)
        residual = obs-newobs
        newH(x) = m.h(x,u) # prev was m.p --> what is p ask Preston
        #@show "here newH"
        H = ForwardDiff.jacobian(newH,x_pred_mean)
        #@show H = ForwardDiff.jacobian!(zeros(11,11),newH,x_pred_mean)
        S = H*cov(x_pred)*H' + R
        tol = sqrt(eps(real(float(one(eltype(S))))))
        K = cov(x_pred)*H'*pinv(S, tol);
        mean_new = mean(x_pred)+ K*residual
        Pnew = (eye(size(cov(x_new),1))-K*H)*cov(x_pred)*(eye(size(cov(x_new),1))-K*H)'+K*R*K';
        cov_new = (1/2)*(Pnew+Pnew')
        cov_new = nearestSPD(cov_new)
        x_new = MvNormal(mean_new[:,1],cov_new)
    end
    return x_new
end
