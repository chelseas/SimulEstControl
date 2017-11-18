using TensorFlow
using Distributions
#using Plots
#plotly()
#using # change this later so don't always have to load
include("SSM.jl")
function nearestSPD(A::Matrix{Float64})
    #Ahat and mineig are any
    #Ahat::Array{Float64,2}
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
function ukf(m::NonLinearSSM, # NL SSM
            z::Vector{Float64}, # obs
            x0::MvNormal,
            Q::Array{Float64,2}, # process noise
            R:: Array{Float64,2}, # measurement noise
            u::Array=zeros(m.nu,size(z,2)))

    # Aliases (notations from UKF papers)
    α = 1e-3
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
## Data load from the MPC files of fully observable stuff --> append reward to states
# Then can take batches of longer numbers of these samples to try and learn things temporally
#include("RNN_Load_Data.jl")
# define buffer and training/testing protocol
epsilon = 0.5 # percent chance of using UKF update to train
processNoise = 0.0001
paramNoise = 0.1
measNoise = 0.0000001
rollout = "train" # train/test mode
initStates = ones(11)
initBelief = MvNormal(ones(11),paramNoise*eye(11))
deltaT = 0.1
fRange = 1.0
ssm = build2DSSM(deltaT)
w = MvNormal(zeros(11),processNoise*eye(11))
v = MvNormal(zeros(ssm.ny),processNoise*eye(ssm.ny))
uDist = MvNormal(zeros(ssm.nu),fRange*eye(ssm.nu)) # call rand to get action
buffer = zeros(50,1,12) # append data --> shape: num_data_samples x seq length (1) x vals in seq (12) # store the labels here too for ease
#=
x_load = zeros(50,1,7)
x_load[:,1,1] = [-188.405; -207.147; -225.314; -228.13; -225.361; -230.68; -219.643; -216.117; -225.683; -196.77; -196.71; -210.041; -158.027; -110.698; -124.069; -116.369; -111.041; -74.6158; -112.414; -136.997; -93.9838; -104.734; -118.439; -90.0561; -53.3899; -59.6105; -54.3256; -27.2777; -50.6422; -37.878; -64.2447; -74.0359; -35.4301; -53.9384; -66.8392; -64.7784; -89.8987; -75.1441; -54.132; -52.6738; -78.8336; -85.7546; -121.092; -128.434; -89.4971; -55.9082; -61.7441; -70.9556; -49.0161; -48.9314]

x_load[:,1,2] = [1.0; 0.382299; 0.233723; 0.0325173; -0.46548; -0.63464; -0.455463; -0.469824; -0.465615; 0.10037; 0.293785; 0.538586; 0.557735; 0.858735; 0.999313; 0.864984; -0.206749; 0.314334; -0.131724; 0.588762; 1.2299; 1.11887; 1.90298; 2.01115; 2.17175; 1.95352; 0.888807; 0.116846; 0.112752; 0.972924; -0.332286; -1.45459; -1.6631; -0.229132; 1.44944; -0.325433; -1.75153; -3.45524; -4.38371; -2.57186; -0.663534; 1.07375; 0.255451; 2.37431; 2.18305; 1.45591; -0.0113829; 1.077; 3.0532; 2.7759]

x_load[:,1,3] = [1.0; 0.0864947; -0.0589909; -0.186533; -0.936007; -1.13451; -1.04974; -1.28852; -1.0838; -0.998939; -0.821049; -1.14827; -1.20911; -1.27459; -1.11703; -1.10322; -1.04432; -1.10932; -0.922085; -1.64673; -2.09388; -2.54904; -2.73958; -2.59201; -1.83216; -1.6292; -1.52576; -1.7275; -0.0328009; -0.0100003; -0.139072; 1.97557; 2.16978; -3.34431; -0.115998; 1.60874; 1.30269; 0.0340299; 0.688406; 1.20974; -4.31977; 1.10823; 1.11137; 1.19411; -0.276559; -0.756467; -0.917661; -1.72556; 0.165774; 1.25878]

x_load[:,1,4] = [1.0; 1.21287; 1.6936; 1.59286; 1.1819; 1.21223; 0.494334; -0.17648; -0.253557; -0.528416; -1.13133; -1.65705; -2.48091; -3.6191; -3.2671; -3.86379; -3.21842; -3.08012; -2.36943; -1.73425; -1.17727; -0.498221; -0.309571; -0.554822; -0.710572; -0.256133; -0.225948; 0.365826; 0.722176; 0.401894; 0.810244; 0.836556; 0.212479; -0.51852; 0.318506; 1.42899; 1.36118; 0.695351; 1.02371; -0.140361; -0.853773; -1.20132; -0.5617; -1.65563; -1.45494; -1.80303; -0.422688; -1.05825; -0.91769; -0.369672]

x_load[:,1,5] = [1.0; 0.733679; 0.638695; 0.433168; 0.203206; 0.173637; 0.144742; 0.099469; -0.280601; -0.257686; -0.378594; -0.538351; -0.266356; -0.255153; -0.302377; 0.010582; -0.193033; -0.0332067; -0.261989; -0.534466; -0.543622; -0.712946; -0.683644; -0.533357; -0.438569; -0.224379; -0.0596363; -0.0481628; -0.251959; 0.141473; 0.224016; 0.458022; 0.137568; -0.223421; 0.0967643; 0.350396; 0.899257; 0.973621; 0.504155; 0.106242; -0.0940919; -0.0477589; -0.532099; -0.689696; -0.462252; -0.0937251; -0.317686; -0.857652; -0.587938; -0.608937]

x_load[:,1,6] = [1.0; 1.19512; 1.29381; 0.980067; 1.20495; 0.924974; 0.716337; 0.591597; 0.83706; 0.664837; 0.633967; 0.732687; 0.745362; 0.459951; 0.595186; 0.708311; 0.743279; 0.520052; 0.946284; 1.23961; 0.925082; 0.876234; 0.766366; 0.507655; 0.434818; 0.503857; 0.451471; 0.123245; 0.0606142; -0.0816999; -0.398954; -0.276638; 0.165998; 0.0563362; -0.429387; -0.401328; -0.0188878; -0.0963283; -0.190894; 0.340151; 0.0643777; -0.197148; -0.146166; -0.0705013; 0.182257; 0.336656; 0.567668; 0.0831186; -0.190924; -0.124214]

x_load[:,1,7] = [1.0; 1.45387; 1.81166; 2.38831; 2.58339; 3.24692; 3.26956; 3.208; 2.88506; 2.75302; 2.65801; 2.65954; 1.87333; 1.46431; 1.36094; 1.3234; 1.00212; 0.509153; 0.2694; 0.191964; 0.288917; 0.235955; 0.550407; 0.479121; 0.0971822; 0.216095; -0.206984; -0.212785; -0.536986; -0.272247; 0.0136971; 0.44974; 0.0825745; -0.156234; -0.29929; -0.0737876; -0.444115; -0.162097; 0.0543346; 0.356829; 0.936786; 1.14601; 1.188; 1.25451; 0.844953; 0.193436; 0.0778506; -0.151558; -0.125179; -0.219052]

y_load = [1.0 1.0 1.0 1.0 1.0; 1.15928 1.16898 1.27528 0.234739 1.39996; 1.33394 1.00423 1.021 0.287505 1.28597; 1.40778 1.09204 0.839529 0.537439 1.42721; 1.79174 0.989109 0.854149 0.544154 0.939711; 1.41003 0.7762 0.912664 0.609729 1.13671; 1.23192 0.1 1.19263 0.330984 0.931277; 1.47336 0.1 1.28821 0.451259 0.999728; 1.45841 0.248125 1.12818 0.575848 0.528274; 1.64252 0.1 0.947316 0.1 0.216166; 1.38776 0.238987 1.0308 0.178498 0.272889; 1.4537 0.376313 0.799869 0.196134 0.748183; 1.34971 0.315615 0.524306 0.287648 0.409029; 1.19442 0.60203 0.659862 0.185322 0.588656; 1.0391 0.917197 0.602053 0.1 0.665201; 0.822938 0.860349 0.800759 0.1 0.484326; 0.866283 0.682047 1.16675 0.160004 0.653711; 1.00916 0.570999 1.16004 0.1 0.228759; 0.832908 0.782209 1.38825 0.1 0.12479; 0.683842 0.316733 1.30175 0.152946 0.1; 0.434513 0.1 1.11343 0.219664 0.1; 0.645507 0.101469 1.06957 0.1 0.1; 0.825779 0.1 0.989413 0.1 0.1; 0.747493 0.215913 1.16338 0.125682 0.1; 0.446889 0.158461 0.954324 0.372936 0.188738; 0.524743 0.252201 1.05813 0.657444 0.1; 0.411703 0.164009 0.775802 0.583411 0.1; 0.242333 0.1 1.56963 0.471013 0.139891; 0.372798 0.111366 1.43131 0.795958 0.292544; 0.404717 0.1 1.2541 0.62893 0.645365; 0.241136 0.460076 1.25442 0.949733 0.714477; 0.407819 0.241096 1.78801 0.835286 0.571704; 0.1 0.1 1.60341 1.15841 0.896197; 0.15066 0.171292 1.57563 1.01197 1.3371; 0.368643 0.345964 1.12668 1.29603 1.22217; 0.259513 0.382278 1.00214 1.11554 1.11317; 0.25469 0.253676 0.781182 1.49205 1.07865; 0.192358 0.495329 0.966241 1.30468 1.27281; 0.11156 0.575943 1.01254 1.46437 1.29342; 0.1 0.818782 1.29321 0.828828 1.39489; 0.1 0.66898 1.3938 0.924103 1.47402; 0.1 0.838893 1.35392 1.10936 1.62434; 0.214114 1.03838 1.28226 1.39034 1.62046; 0.50287 0.896289 1.37553 1.62891 1.53941; 0.638938 0.824124 1.15066 1.74753 1.84197; 0.351191 0.760627 1.02291 2.10503 1.70062; 0.275873 0.311766 1.47943 2.18637 1.6935; 0.155332 0.1 1.45452 1.76526 1.23379; 0.1 0.1 1.3118 1.64463 0.731031; 0.1 0.1 1.71642 1.18481 0.625559]

numsteps_test = 10
x = x_load[1:numsteps_test,:,:]
#@show x
y = y_load[1:numsteps_test,:]
=#
## Initializing TF setup and RNN
# Build the model
sess = Session(Graph())
# Data Placeholders
function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end
srand(13) # seeding
#const x = randn(100,1,6) # num_samples of data x length of sequence (how many feat vectors to pass in for 1 prediction) x num_features (length of the vector)

const samples = 50#size(x)[1] # total number of data points
const S = 1#size(x)[2] # sequences per prediction
const N = 7#size(x)[3] # input layer dims
#const y = cos.([x[:,1,1:end-2] sum(x[:,1,end-1:end],2)]) # need num_samples x output
const O = 5#size(y)[2] # output layer dims
const H = 128 # hidden layer dims
X = placeholder(Float64, shape=[-1, S, N]) # here batch_size = 1
Y_obs = placeholder(Float64, shape=[-1, O])

# could define a class for DQN here if making multiple
#abstract type Network end
#mutable DRQN <: Network
#    function RNN(N::)
#end

## Initialize Network
## switch to dynamic rnn at some point--> faster
W = weight_variable([H,O]) # hidden x output
B = weight_variable([O])
Hs, states = nn.rnn(nn.rnn_cell.LSTMCell(H), X; dtype=Float32)
#@show get_shape.(Hs) # Tensor containing 1 tensor of size samples x hidden
Y=Hs[end]*W+B#nn.softmax(Hs[end]*W+B) # samples x hidden * hidden * output = samples x output
dif_error = squared_difference(Y_obs, Y) # eventually use the Q values and make this TD Error
Loss = reduce_mean(dif_error)#log(Y).*Y_obs) # cross entropy
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver = train.Saver()

# Run training
run(sess, global_variables_initializer())
save_dir = "RNN_saved"
try mkdir(save_dir)
end
checkpoint_path = abspath(save_dir)
info("Checkpoint files saved in $checkpoint_path")
temp = 0
epoch_limits =100
y_train = zeros(50,5)
losses_vec = zeros(epoch_limits)
for epoch in 1:epoch_limits
    x = rand(initBelief)
    x_ = rand(initBelief)
    b = initBelief
    for i in 1:50 # steps
      u = rand(uDist)
      x_ = ssm.f(x,u)+rand(w)
      b = ukf(ssm,x[1:ssm.ny],b,cov(w),cov(v),u)
      rew = sum([x;u])
      buffer[i,:,:] = [rew; x]
      x = x_
    end
    x_train = buffer[:,:,1:7]
    y_train[:,:] = buffer[:,:,8:end]
    cur_loss, _, var = run(sess, [Loss, minimize_op, Y], Dict(X=>x_train, Y_obs=>y_train))
    losses_vec[epoch] = cur_loss
    if epoch%1 == 0 || epoch == 1
        println(@sprintf("%.2f", cur_loss))
    end
    if epoch == epoch_limits
        temp = var
        train.save(saver, sess, joinpath(checkpoint_path, "RNN_RandU.jld"))
    end
    #println(@sprintf(" ex var: %.2f.",var)
end
@show losses_vec
#plot(losses_vec)
# visualize()
#=
# Test the output
sess2 = Session(Graph())

X = placeholder(Float64, shape=[samples, S, N]) # here batch_size = 1
Y_obs = placeholder(Float64, shape=[samples, O])
W = weight_variable([H,O]) # hidden x output
B = weight_variable([O])
Hs, states = nn.dynamic_rnn(nn.rnn_cell.LSTMCell(H), X; dtype=Float32)
#@show get_shape.(Hs) # Tensor containing 1 tensor of size samples x hidden
Y=Hs[end]*W+B#nn.softmax(Hs[end]*W+B) # samples x hidden * hidden * output = samples x output
dif_error = squared_difference(Y_obs, Y) # eventually use the Q values and make this TD Error
Loss = reduce_mean(dif_error)#log(Y).*Y_obs) # cross entropy
optimizer = train.AdamOptimizer()
minimize_op = train.minimize(optimizer, Loss)
saver2 = train.Saver()
@show checkpoint_path
train.restore(saver2,sess2,string(checkpoint_path,"/RNN-500.jld"))
pred = run(sess2, [Y], Dict(X=>x, Y_obs=>y))
@show pred
=#
