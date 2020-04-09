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

# Train with mini-batch simultaneous perturbation stochastic approximation (SPSA)
maxIter = 150000
stepSize = 1.25e-3
c = 5e-2
batchSize = 20
gamma = 0.5
dropStepSizeIter = 40000
dropCIter = 15000
incBatchFactor = 2
incBatchIter = 100000
printIter = 2000
for iter in 1:maxIter

	# Generate perturbations from Rademacher distribution
	delta = 2*round.(rand(nParams,1)) .- 1

	# Evaluate mini-batch losses	
	batchIndices = sample(1:n,batchSize,replace=false)
	w_plus = w + c*delta
	w_minus = w - c*delta

	(f_plus_sum,~) = NeuralNetMulti_backprop(w_plus,X[batchIndices[1],:],Y[batchIndices[1],:],k,nHidden)	
	for i in 2:batchSize
		(f_plus,~) = NeuralNetMulti_backprop(w_plus,X[batchIndices[i],:],Y[batchIndices[i],:],k,nHidden)
		f_plus_sum += f_plus
	end
	f_plus_av = (1/batchSize)*f_plus_sum

	(f_minus_sum,~) = NeuralNetMulti_backprop(w_minus,X[batchIndices[1],:],Y[batchIndices[1],:],k,nHidden)	
	for i in 2:batchSize
		(f_minus,~) = NeuralNetMulti_backprop(w_minus,X[batchIndices[i],:],Y[batchIndices[i],:],k,nHidden)
		f_minus_sum += f_minus
	end
	f_minus_av = (1/batchSize)*f_minus_sum

	# Calculate gradient estimate and take a step
	g_hat = (f_plus_av - f_minus_av)./(2*c*delta)
	global w = w - stepSize*g_hat

	# Print some information every few iterations
	if (mod(iter,printIter) == 0)
		ypred = NeuralNet_predict(w,X,k,nHidden)
		yhat = NeuralNet_predict(w,Xtest,k,nHidden)
		train_error = sum(ypred .!= y)/n; test_error = sum(yhat .!= ytest)/t
		@printf("Training iteration = %d, train error = %0.4f, test error = %0.4f, step size = %f, batch size = %d\n",iter,train_error,test_error,stepSize,batchSize)
	end

	# Drop step size (learning rate) according to schedule
	if (mod(iter,dropStepSizeIter) == 0)
		global stepSize = gamma*stepSize
	end

	# Drop perturbation size c according to schedule
	if (mod(iter,dropCIter) == 0)
		global c = 0.1*c
	end

	# Increase batch size according to schedule
	if (mod(iter,incBatchIter) == 0)
		global batchSize = incBatchFactor*batchSize
	end
end


