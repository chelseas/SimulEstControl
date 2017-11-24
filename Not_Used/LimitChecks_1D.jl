<### control limiting
control_limit = 1.05*fRange
control_lim = fRange

function control_check(u::Array{Float64,1}, x::Array{Float64,1}, flag::Bool)
    if(isnan(u[1]))
      u[1] = 0.0
    end

    if(u[1] > control_limit)
        u[1] = control_lim
    elseif(u[1] < -control_limit)
        u[1] = -control_lim
    end

    return u
end

### debu[g checker for the states
function state_check(x::Array{Float64,1}, flag::Bool)
    if(x[3] < state_init*state_min_tol)
        x[3] = state_init*state_min_tol
        if(flag)
            @show "mass here"
        end
    end
    return x
end

### debug checker for the estimated parameters
function est_check(xNew::Array{Float64,1}, flag::Bool)
    if(xNew[3] < state_init*state_min_tol)
        xNew[3] = state_init*state_min_tol
        if(flag)
            @show "state bounded"
        end
    end

    return xNew
end
