### control limiting
control_limit = 1.05*fRange
control_lim = fRange

function control_check(u::Array{Float64,1}, x::Array{Float64,1}, flag::Bool)
    if isnan(u[1])
        if(x[4] < 0)
            u[1] = control_lim
        else
            u[1] = -control_lim
        end
    elseif(u[1] > control_limit)
        u[1] = control_lim
    elseif(u[1] < -control_limit)
        if(flag)
            @show u
        end
        u[1] = -control_lim
    end

    if isnan(u[2])
        if(x[5] < 0)
            u[2] = control_lim
        else
            u[2] = -control_lim
        end
    elseif(u[2] > control_limit)
        u[2] = control_lim
    elseif(u[2] < -control_limit)
        if(flag)
            @show u
        end
        u[2] = -control_lim
    end

    if isnan(u[3])
        if(x[6] < 0)
            u[3] = control_lim
        else
            u[3] = -control_lim
        end
    elseif(u[3] > control_limit)
        u[3] = control_lim
    elseif(u[3] < -control_limit)
        if(flag)
            @show u
        end
        u[3] = -control_lim
    end

    return u
end

### debu[g checker for the states
function state_check(x::Array{Float64,1}, flag::Bool)
    if(x[7] < state_init*state_min_tol)
        x[7] = state_init*state_min_tol
        if(flag)
            @show "mass here"
        end
    end
    if(x[8] < state_init*state_min_tol)
        x[8] = state_init*state_min_tol
        if(flag)
            @show "Inertia here"
        end
    end
    if(x[9] < state_init*state_min_tol)
        x[9] = state_init*state_min_tol
        if(flag)
            @show "Fr here"
        end
    end
    if(x[10] < state_init*state_min_tol)
        x[10] = state_init*state_min_tol
        if(flag)
            @show "Rx here"
        end
    end
    if(x[11] < state_init*state_min_tol)
        x[11] = state_init*state_min_tol
        if(flag)
            @show "Ry here"
        end
    end
    return x
end

### debug checker for the estimated parameters
function est_check(xNew::Array{Float64,1}, flag::Bool)
    bound = 100000.0
    if(xNew[1] < -bound)
        xNew[1] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[1] > bound)
        xNew[1] = bound
    end
    if(xNew[2] < -bound)
        xNew[2] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[2] > bound)
        xNew[2] = bound
    end
    if(xNew[3] < -bound)
        xNew[3] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[3] > bound)
        xNew[3] = bound
    end
    if(xNew[4] < -bound)
        xNew[4] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[4] > bound)
        xNew[4] = bound
    end
    if(xNew[5] < -bound)
        xNew[5] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[5] > bound)
        xNew[5] = bound
    end
    if(xNew[6] < -bound)
        xNew[6] = -bound
        if(flag)
            @show "state bounded"
        end
    elseif(xNew[6] > bound)
        xNew[6] = bound
    end

    if(xNew[7] < state_init*state_min_tol)
        xNew[7] = state_init*state_min_tol
        if(flag)
            @show "7 here"
        end
    elseif(xNew[7] > bound)
        xNew[7] = bound
    end
    if(xNew[8] < state_init*state_min_tol)
        xNew[8] = state_init*state_min_tol
        if(flag)
            @show "8 here"
        end
    elseif(xNew[8] > bound)
        xNew[8] = bound
    end
    if(xNew[9] < state_init*state_min_tol)
        xNew[9] = state_init*state_min_tol
        if(flag)
            @show "9 here"
        end
    elseif(xNew[9] > bound)
        xNew[9] = bound
    end

    if(xNew[10] < -bound)
        xNew[10] = -bound
        if(flag)
            @show "10 here"
        end
    elseif(xNew[10] > bound)
        xNew[10] = bound
    end
    if(xNew[11] < -bound)
        xNew[11] = -bound
        if(flag)
            @show "11 here"
        end
    elseif(xNew[11] > bound)
        xNew[11] = bound
    end

    return xNew
end
