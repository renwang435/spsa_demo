using Distributions
using JLD
using Printf
using PyPlot
using Statistics
include("misc.jl")

# Load X and y variable
data = load("uspsData.jld")
(X,y,Xtest,ytest) = (data["X"],data["y"],data["Xtest"],data["ytest"])
(n,d) = size(X)
t = size(Xtest,1)

# Standardize columns and add bias variable to input layer
(X,mu,sigma) = standardizeCols(X)
X = [ones(n,1) X]
d += 1

# Apply the same transformation to test data
Xtest = standardizeCols(Xtest,mu=mu,sigma=sigma)
Xtest = [ones(t,1) Xtest]

# Let 'k' be the number of classes, and 'Y' be a matrix of binary labels
k = maximum(y)
Y = zeros(n,k)
for i in 1:n
	Y[i,y[i]] = 1
end

# Choose neural network structure and randomly initialize weights
include("NeuralNetReg.jl")
nHidden = [10]
@show nHidden
nParams = NeuralNetMulti_nParams(d,k,nHidden)
w = randn(nParams,1)
@show size(w)

# Train with mini-batch stochastic gradient descent
maxIter = 80000
stepSize = 5.0e-3
batchSize = 20
gamma = 0.5
dropStepSizeIter = maxIter
incBatchFactor = 2
incBatchIter = 100000
printIter = 1000

train_errors = zeros(0)
test_errors = zeros(0)
iters = zeros(0)

for iter in 1:maxIter

	# Calculate mini-batch gradients
	batchIndices = sample(1:n,batchSize,replace=false)
	(f_sum,g_sum) = NeuralNetMulti_backprop(w,X[batchIndices[1],:],Y[batchIndices[1],:],k,nHidden)
	for i in 2:batchSize
		(f,g) = NeuralNetMulti_backprop(w,X[batchIndices[i],:],Y[batchIndices[i],:],k,nHidden)
		f_sum += f; g_sum += g
	end
	f_av = (1/batchSize)*f_sum; g_av = (1/batchSize).*g_sum

	# SGD update
	global w = w - stepSize*g_av

	# Print some information every few iterations
	if (mod(iter,printIter) == 0)
		ypred = NeuralNet_predict(w,X,k,nHidden)
		yhat = NeuralNet_predict(w,Xtest,k,nHidden)
		train_error = sum(ypred .!= y)/n; test_error = sum(yhat .!= ytest)/t
		@printf("Training iteration = %d, train error = %0.4f, test error = %0.4f, step size = %f, batch size = %d\n",iter,train_error,test_error,stepSize,batchSize)
		# Store current train/test errors and iter number for plotting.
		append!(iters, iter)
		append!(train_errors, train_error)
		append!(test_errors, test_error)
	end

	# Drop step size (learning rate) according to schedule
	if (mod(iter,dropStepSizeIter) == 0)
		global stepSize = gamma*stepSize
	end

	# Increase batch size according to schedule
	if (mod(iter,incBatchIter) == 0)
		global batchSize = incBatchFactor*batchSize
	end
end

# Save results.
save("results/sgd_results.jld", "iters", iters, "train_errors", train_errors, "test_errors", test_errors)
# Load and plot results.
results = load("results/sgd_results.jld")
iters = results["iters"]
train_errors = results["train_errors"]
test_errors = results["test_errors"]
fig = figure(figsize=(10.00, 8.00))
ax = fig.add_subplot(111)
ax.plot(iters, train_errors, ".")
ax.set_xlabel("Iterations")
ax.set_ylabel("Training Error")
ax.set_title("Training Performance of SGD")
plt.show()
