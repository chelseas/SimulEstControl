### control limiting
control_limit = fRange+10.0
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
    if(x[7] < 1.0)
        x[7] = 1.0
        if(flag)
            @show "mass here"
        end
    end
    if(x[8] < 1.0)
        x[8] = 1.0
        if(flag)
            @show "Inertia here"
        end
    end
    if(x[9] < 1.0)
        x[9] = 1.0
        if(flag)
            @show "Fr here"
        end
    end
    if(x[10] < 1.0)
        x[10] = 1.0
        if(flag)
            @show "Rx here"
        end
    end
    if(x[11] < 1.0)
        x[11] = 1.0
        if(flag)
            @show "Ry here"
        end
    end
    return x
end

### debug checker for the estimated parameters
function est_check(xNew::Array{Float64,1}, flag::Bool)
    bound = 100000.0
    upper_b = 1000.0
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

    if(xNew[7] < 1.0)
        xNew[7] = 1.0
        if(flag)
            @show "7 here"
        end
    elseif(xNew[7] > upper_b)
        xNew[7] = upper_b
    end
    if(xNew[8] < 1.0)
        xNew[8] = 1.0
        if(flag)
            @show "8 here"
        end
    elseif(xNew[8] > upper_b)
        xNew[8] = upper_b
    end
    if(xNew[9] < 1.0)
        xNew[9] = 1.0
        if(flag)
            @show "9 here"
        end
    elseif(xNew[9] > upper_b)
        xNew[9] = upper_b
    end

    if(xNew[10] < -50.0)
        xNew[10] = -50.0
        if(flag)
            @show "10 here"
        end
    elseif(xNew[10] > 50.0)
        xNew[10] = 50.0
    end
    if(xNew[11] < -50.0)
        xNew[11] = -50.0
        if(flag)
            @show "11 here"
        end
    elseif(xNew[11] > 50.0)
        xNew[11] = 50.0
    end

    return xNew
end
