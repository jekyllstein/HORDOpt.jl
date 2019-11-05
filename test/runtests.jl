using HORDOpt
using Test
using DecisionTree
using Random
using PrettyTables

Random.seed!(0)

m = 1000
ncols = 14
m2 = floor(Int64, m/2)
X = rand(m, ncols)
yideal = (10 .*sin.(pi.*X[:, 1].*X[:, 2])) .+ 20 .*(X[:, 3] .- 0.5).^2 .+ 10 .*X[:, 4] + 5 .*X[:, 5]
y = yideal .+ randn(m)
#y only depends on X1 to X5, X6 through X10 are random noise variables

#X11 to X14 are correlated with X1 to X4 but no directly part of calculating y
X[:, 11:14] .= X[:, 1:4] .+ 0.01.*randn(m, 4)

randind = shuffle(1:m)
traininds = randind[1:m2]
testinds = randind[m2+1:m]

Xtrain = X[traininds, :]
Xtest = X[testinds, :]
ytrain = y[traininds]
ytest = y[testinds]

getsqerr(y1, y2) = sum((y1 .- y2).^2) / length(y1)

function trainforest(Xtrain, ytrain, Xtest, ytest, params...)
	labels = ytrain
	features = Xtrain
	model = build_forest(ytrain, Xtrain,
	                     params...)
	yforesttest = apply_forest(model, Xtest)
	testerr = getsqerr(yforesttest, ytest)
	yforesttrain = apply_forest(model, Xtrain)
	trainerr = getsqerr(yforesttrain, ytrain)
	if isnan(testerr)
		(Inf, trainerr)
	else
		(testerr, trainerr)
	end
end

pnames = ["n_subfeatures", "n_trees", "partial_sampling", "max_depth", "min_samples_leaf", "min_samples_split", "min_purity_increase"]

opt_params = ( 	(1, ncols),  #n_subfeatures: number of features to consider at random per split (default: -1, sqrt(# features))
			   	(100,), #n_trees
			   	(0.1, 1.0), #partial_sampling
				(-1,), #max_depth
				(1, 10), #min_samples_leaf
				(2,), #min_samples_split
				(0.0,), #min_purity_increase
			)

#convert necessary variables to integers and clamp values in desired range
pconvert = makepconvert(opt_params, minvals = [1, 1, eps(0.0), -1, 1, 1, 0.0], maxvals = [ncols, typemax(Int64), 1.0, -1, typemax(Int64), 2, 0.0])

h = findall(p -> length(p) == 2, opt_params)

#test raw_params conversion from a random vector between 0 and 1
raw_params = Tuple(rand(length(h)))

#test that parameter conversions are correct
params = convert_params(pconvert, raw_params, opt_params, pnames)
@test params[[4, 6, 7]] == (-1, 2, 0.0) #"$(params[[1, 4, 6, 7]]) does not match (-1, -1, 2, 0.0)"
@test typeof(params[1]) <: Integer
@test typeof(params[2]) <: Integer
@test typeof(params[5]) <: Integer

isp = (8, 100, 1.0, -1, 1, 2, 0.0)
(results, resultsdict, cnames, fixednames, fixedparams) = runHORDopt_trials(params -> trainforest(Xtrain, ytrain, Xtest, ytest, params...), opt_params, 50, isp, pnames = pnames, pconvert = pconvert)

output = mapreduce(vcat, results) do a
    [a[1]; collect(a[2]); collect(a[3])]'
end
header = ["Trial"; "Best Test Error"; pnames]

println("Results for holding the followering parameters: $fixednames")
println("Fixed at: $fixedparams")
pretty_table(output, header)

