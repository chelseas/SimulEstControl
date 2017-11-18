## Defines the DRQN behavior to be called at each action step
# use existing RNN and have it first train on a bunch with EKF operating
# then switch over and see if it can predict well -->
# tot_buffer = zeros(numtrials*nSamples,S,N) # store data from all epochs to sample for exp replay
# Build the model
sess = Session(Graph()) #segfault here
const S = 1 # sequences per prediction
const N = ssm.states+1 # input layer dims 6 + rew
const O = ssm.nx - ssm.states # output layer dims of hidden params
const H = 128 # hidden layer dims
buffer = zeros(nSamples,S,N) # to store data from each run
function weight_variable(shape)
    initial = map(Float32, rand(Normal(0, .001), shape...))
    return Variable(initial)
end
# Data Placeholders
X = placeholder(Float64, shape=[-1, S, N]) # here batch_size = 1
Y_obs = placeholder(Float64, shape=[-1, O])
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

# for testing use an if else with the stuff above and first load the weights
