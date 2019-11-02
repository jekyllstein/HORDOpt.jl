"""
    makepconvert(opt_params; minvals = [typemin(typeof(a[1])) for a in opt_params], maxvals = [typemin(typeof(a[1])) for a in opt_params])

Generate tuple of functions to optionally transform parameter values after being generated.

For elements of `opt_params` that are single values, uses the identity functions.
For elements with a range, applies type conversion and rounding in the case of integers
and optionally clamps output value to a range.  By default it uses the extreme values of
that type.

# Examples
```julia-repl
julia> using HORDOpt
julia> ncols = 14;
julia> opt_params = (  (1, ncols),  #n_subfeatures
                (100,), #n_trees
                (0.1, 1.0), #partial_sampling
                (-1,), #max_depth
                (1, 10), #min_samples_leaf
                (2,), #min_samples_split
                (0.0,), #min_purity_increase
            );
julia> pconvert = makepconvert(opt_params, minvals = [1, 1, eps(0.0), -1, 1, 1, 0.0], maxvals = [ncols, typemax(Int64), 1.0, -1, typemax(Int64), 2, 0.0]);
```
"""
function makepconvert(opt_params; minvals = [typemin(typeof(a[1])) for a in opt_params], maxvals = [typemin(typeof(a[1])) for a in opt_params])
    Tuple([begin
        if length(p) == 2
            T = typeof(p[1])
            f = T <: Integer ? round : identity
            a -> T(clamp(f(a), minvals[i], maxvals[i]))
        else
            identity
        end
    end
    for (i, p) in enumerate(opt_params)])
end

"""
    convert_params(pconv, raw_params, opt_params, pnames::Vector{T}) where T <: AbstractString

Creates a tuple of parameters to use as arguments in the optimization function by rescaling the values
in raw_params from a 0-1 range into the range given in `opt_params`.

!!! note
    For parameters that remain fixed, they are specified by applying the appropriate
    pconv function to the single value in opt_params.  All other values are first
    rescaled from the raw value into the range and then transformed by pconv.

# Examples
```julia-repl
julia> using HORDOpt
julia> pnames = ["n_subfeatures", "n_trees", "partial_sampling", "max_depth", "min_samples_leaf", "min_samples_split", "min_purity_increase"];
julia> ncols = 14;
julia> opt_params = (   (1, ncols),  #n_subfeatures
                        (100,), #n_trees
                        (0.1, 1.0), #partial_sampling
                        (-1,), #max_depth
                        (1, 10), #min_samples_leaf
                        (2,), #min_samples_split
                        (0.0,), #min_purity_increase
                   );
julia> pconvert = makepconvert(opt_params, minvals = [1, 1, eps(0.0), -1, 1, 1, 0.0], maxvals = [ncols, typemax(Int64), 1.0, -1, typemax(Int64), 2, 0.0]);
julia> h = findall(p -> length(p) == 2, opt_params);
julia> raw_params = Tuple(rand(length(h)))
(0.18248479399609585, 0.03425670579069151, 0.4141927499235867)
julia> params = convert_params(pconvert, raw_params, opt_params, pnames)

Using the following fixed hyper parameters : n_trees = 100, max_depth = -1, min_samples_split = 2, min_purity_increase = 0.0,
Setting the following hyper parameters : n_subfeatures = 3, partial_sampling = 0.13083103521162237, min_samples_leaf = 5,

(3, 100, 0.13083103521162237, -1, 5, 2, 0.0)
```
"""
function convert_params(pconv, raw_params, opt_params, pnames::Vector{T}) where T <: AbstractString
    #vector of indices that contain two values => these parameters will be tuned
    h = findall(p -> length(p) == 2, opt_params)
    
    #the number of raw params should match the number of parameters with a range
    @assert length(raw_params) == length(h) 
    @assert length(pconv) == length(opt_params)

    paramsdict = Dict(zip(h, raw_params))

    #scaledParams = map((a, b) -> mapRange(a, b[1], b[2]), rawParams, opt_params[h])

    #vector of remaining indices not being tuned
    ih = setdiff(eachindex(opt_params), h)
    
    #generate full set of training parameters to be used properly converted
    params = map(opt_params, pconv, Tuple(eachindex(opt_params))) do op, pc, p
        if in(p, h)
            #rescale parameter from 0 to 1 range into specified range
            scaledparam = maprange(paramsdict[p], op[1], op[2])
            pc(scaledparam)
        else
            pc(op[1])
        end
    end

    println()
    if !isempty(ih)
        println(string("Using the following fixed hyper parameters : ", mapreduce(i -> string(pnames[i], " = ", params[i], ", "), (a, b) -> string(a, b), ih)))
    end
    println(string("Setting the following hyper parameters : ", mapreduce(i -> string(pnames[i], " = ", params[i], ", "), (a, b) -> string(a, b), h)))
    println()
    return params 
end



function run_opt_func(optfunc::Function, params, resultsdict)
    out = if haskey(resultsdict, params)
        println("Using results dictionary instead of new function evaluation")
        resultsdict[params]
    else
        (optfunc(params)...,)
    end 
    
    push!(resultsdict, params => out)
    return out
end

function run_opt_func(optfunc::Function, params)
    out = (optfunc(params)...,)
    resultsdict = Dict(params => out)
    (out, resultsdict)
end



###########################################Main Algorithm######################################################
"""
    run_HORDopt(optfunc::Function, opt_params::NTuple{U, N}, trialid, nmax, isp = []; resultsdict = (), pnames = ["Parameter \$n" for n in 1:length(opt_params)], pconvert::NTuple{T, N} = map(identity, opt_params)) where T <: Function where U <: Real where N

Runs hyperparameter optimization algorithm based on dynamic search 
with and RBF surrogate function.  Given an optimization function and 
set of tunable parameters, iterates through trials attempting to minimize 
the error objective.

The return type is a tuple of 4 vectors with results for each trial and a 
dictionary of results for a given set of parameters.
1. Objective error 
2. Parameters used
3. Other function outputs
4. Parameter vectors scaled into 0-1 range
5. Dictionary of optfunc outputs for given parameter inputs

See `runtests.jl` for a complete example using DecisionTree.jl optimizing the parameters
for a random forest.

!!! note
    `optfunc` must be a function that takes as input the number of parameters 
    contained in opt_params as single values.  Its output must be one or several 
    values with the first value being the objective to be minimized.
    
    `opt_params` is a tuple of tuples that contain either the single parameter 
    to remain fixed or a range for a parameter to vary over.  The values must be 
    finite to allow valid steps through the parameter space.
    
    `pconvert` is a tuple of functions the same length as opt_params that optionally 
    transform the values in the given range.  For example, a range of 0,1 can be transformed 
    into 0 to Inf with f(x) = 1/(x - 1) + 1 

# Examples
```julia-repl
using HORDOpt
julia> opt_params = ((0.0, 1.0), (0.0,), (0.0, 1.0))
((0.0, 1.0), (0.0,), (0.0, 1.0))
julia> pconvert = map(a -> b -> clamp(b, 0.0, 1.0-eps(1.0)), opt_params)
(getfield(Main, Symbol("##8#10"))(), getfield(Main, Symbol("##8#10"))(), getfield(Main, Symbol("##8#10"))())
julia> (errs, params, outputs, xs, resultsdict) = HORDOpt(optfunc, opt_params, pconvert, trialid, nmax, pconvert = pconvert)
```
"""
function run_HORDopt(optfunc::Function, opt_params, trialid, nmax, isp = []; resultsdict = (), pnames = ["Parameter $n" for n in 1:length(opt_params)], pconvert = map(identity, opt_params))
    #vector of indices that contain two values => these parameters will be tuned
    h = findall(p -> length(p) == 2, opt_params)

    #vector of remaining indices not being tuned
    ih = setdiff(eachindex(opt_params), h)

    #scale the isp to the 0 to 1 X range
    isp_x = if isempty(isp)
        []
    else
        # map((a, b) -> mapRangeInv(a, b[1], b[2]), isp[h],  opt_params[h])
        [map_range_inv(isp[i], opt_params[i][1], opt_params[i][2]) for i in h]
    end

    xs = Vector{Vector{Float64}}()
    errs = Vector{Float64}()
    outputs = Vector()
    params = Vector{Tuple}()
    #adding previous values from resultsdict to xs
    for r in resultsdict
        ps = r[1]
        x = [map_range_inv(ps[i], opt_params[j][1], opt_params[j][2]) for (i, j) in enumerate(h)]
        out = run_opt_func(optfunc, ps, resultsdict)

        #add new params to list
        push!(params, ps)
        #extract current training errors which we are trying to minimize
        push!(errs, out[1])
        #extract the other output variables 
        if length(out) > 1
            push!(outputs, out[2])
        else
            push!(outputs, ())
        end
        push!(xs, x)
    end
    indcorrect = length(errs)

    if !isempty(isp_x)
        println("Prepending initial starting point to parameter vectors")
        push!(xs, isp_x)
    end

    println()
    println(string("On trial ", trialid, " tuning the following hyperparameters: ", mapreduce(a -> string(pnames[a], ", "), (a, b) -> string(a, b), h)))
    if !isempty(ih)
        println(string("Keeping the folowing hyperparameters fixed: ", mapreduce(a -> string(pnames[a], " = ", pconvert[a](opt_params[a][1]), ", "), (a, b) -> string(a, b), ih)))
    end
    println()

    #string that contains the fixed values for the HORD training
    ihnames = if isempty(ih)
        ""
    else
        mapreduce(a -> string(pconvert[a](opt_params[a][1]), "_", pnames[a], "_"), (a, b) -> string(a, b), ih)
    end

    #--------Predefined Variables----------------------
    #number of hyperparameters to tune
    d = length(h)

    #initial number of configurations to try
    n0 = 2*(d + 1)

    #number of candidate points to consider each step
    m = 100*d

    #weight balance
    w = 0.3

    #variance for weight perterbations
    varn = 0.2

    #number of concecutive failed iterations
    tfail = 0

    #number of concecutive successful iterations
    tsucc = 0

    #generate a latin hypercube sample of n0 points using and interval of 0 to 1
    #divided into n0 sections
    paramvec = LinRange(0, 1, n0)

    ##----------------------ALGORITHM INITIALIZATION------------------------------------
    #for each coordinate generate a list of n0 unique values to sample from -1 to 1
    println(string("Generating ", n0, " initial parameter vectors"))
    Random.seed!(trialid)
    samplevecs = map(a -> randperm(n0), 1:d)
    for i in 1:n0
        #take the ith element of each sample vec so that once an element has been used
        #it will not appear in any other point
        x = map(1:d) do j
            v = samplevecs[j]   
            paramvec[v[i]]
        end
        push!(xs, x)
    end

    n0 = length(xs) - indcorrect

    println()
    println("Performing initial $n0 optimizations")
    println()
    #generate initial results of Xs parameter samples
    
    println("Starting initial point 1 of $n0")
    p1 = convert_params(pconvert, xs[1+indcorrect], opt_params, pnames)
    if isempty(resultsdict)
        (output1, resultsdict) = run_opt_func(optfunc, p1) 
    else
        output1 = run_opt_func(optfunc, p1, resultsdict)
    end
    err1 = output1[1]
    if length(output1) > 1
        otheroutput1 = output1[2]
    else
        otheroutput1 = ()
    end
    push!(errs, err1)
    push!(params, p1)
    push!(outputs, otheroutput1)
    
    for (i, x) in enumerate(view(xs, indcorrect+2:indcorrect+n0))
        println("------------------------------------------------")
        println("Starting initial point $i of $n0")
        println("------------------------------------------------")
        ps = convert_params(pconvert, x, opt_params, pnames)
        out = run_opt_func(optfunc, ps, resultsdict)

        #add new params to list
        push!(params, ps)
        #extract current training errors which we are trying to minimize
        push!(errs, out[1])
        #extract the other output variables 
        if length(out) > 1
            push!(outputs, out[2])
        else
            push!(outputs, ())
        end
    end

 

    #initial number of configurations
    n = n0

    # set dummy xnew point
    xnew = xs[1]
    failcounter = 0

    ##---------------------ALGORITHM LOOP----------------------------------------------
    while (n < nmax) & (tfail < max(5, d)*3) & (failcounter < 10) & (varn > 1e-6)
        println()
        println(string("Updating surrogate model on iteration ", n, " out of ", nmax))
        println()

        phi = calc_phi_mat(xs)
        p = formP(xs)

        mat1 = [phi p; p' zeros(d+1, d+1)]
        vec = [errs; zeros(d+1)]

        #interpolated paramters
        c = pinv(mat1) * vec

        lambda = c[1:length(xs)]
        b = c[length(xs)+1:end-1]
        a = c[end]

        (errbest, indbest) = findmin(errs)
        xbest = xs[indbest]

        indstr = indbest - indcorrect < 1 ? "a previous trial iteration" : "iteration $(indbest-indcorrect)" 
        println()
        println(string("Current lowest error is ", errbest, " from $indstr using the following configuration:"))
        if !isempty(ih)
            println(string("Fixed hyper parameters:", mapreduce(i -> string(pnames[i], " = ", params[indbest][i], ", "), (a, b) -> string(a, b), ih)))
        end
        println(string("Tuned hyper parameters:", mapreduce(i -> string(pnames[i], " = ", params[indbest][i], ", "), (a, b) -> string(a, b), h)))
        println()

        phi_n = calc_phiN(d, n, n0, nmax)

        validnewpoint = false
        failcounter = 0
        while !validnewpoint & (failcounter < 10)

            #calculate candidate points
            candidatepoints = fill_omegaN(xbest, phi_n, varn)

            #calculate surrogate values for each candidate point
            surrogatevalues = map(t -> surrogate_model(t, xs, lambda, b, a), candidatepoints)
            smax = maximum(surrogatevalues)
            smin = minimum(surrogatevalues)

            #calculate distances from the previously evaluated points for each surrogate point and select the minimum distance
            deltas = map(candidatepoints) do t
                delts = map(xs) do x
                    norm(t .- x)
                end

                minimum(delts)
            end

            deltamax = maximum(deltas)
            deltamin = minimum(deltas)

            #estimated value scores for candidate points
            value_estimates = if smax == smin
                ones(length(candidatepoints))
            else
                map(s -> (s - smin)/(smax - smin), surrogatevalues) 
            end

            #distance metric scores for candidate points
            distancemetrics = if deltamax == deltamin
                ones(length(candidatepoints))
            else
                map(d -> (deltamax - d)/(deltamax - deltamin), deltas)
            end

            #final weighted score for candidate points
            score = w*value_estimates .+ (1-w)*distancemetrics

            #cyclically permute through weights
            w = permuteweight(w)

            #select the point that has the lowest score to add as a new configuration
            (bestscore, bestind) = findmin(score)
            xnew = candidatepoints[bestind]

           

            paramsdict = Dict(zip(h, xnew))

          

            #generate full set of training parameters to be used properly converted
            candidateparams = map(eachindex(opt_params)) do p
                if in(p, h)
                    #rescale parameter from 0 to 1 range into specified range
                    scaledparam = maprange(paramsdict[p], opt_params[p][1], opt_params[p][2])
                    pconvert[p](scaledparam)
                else
                    pconvert[p](opt_params[p][1])
                end
            end

            validnewpoint = !in(candidateparams, params) 
            failcounter += 1
        end


        println()
        println("Optimizing with newly selected configuration")
        println()
        #calculate ANN error with new parameter configuration
        # (fNew, paramsNew, outputNew) = f(X, Xtest, Y, Ytest, xnew, pconvert, batchSize, OPT_PARAMS)
        paramsnew = convert_params(pconvert, xnew, opt_params, pnames)
        outnew = run_opt_func(optfunc, paramsnew, resultsdict)
      
        errnew = outnew[1]
        if length(outnew) > 1
            outputnew = outnew[2]
        else
            outputnew = ()
        end
    
        #iterate function evaluation counter
        n += 1

        #update Tsucc, Tfail based on results
        (tsucc, tfail) = if errnew < errbest
            println()
            println(string("New configuration has a new lowest test set error of ", errnew))
            println()
            (tsucc + 1, 0)
        else
            println()
            println(string("New configuration has a worse test set error of ", errnew))
            println()
            (0, tfail + 1)
        end

        #update perturbation variance if needed
        varn = if tsucc >= 3
            min(0.2, varn*2)
        elseif tfail >= max(5, d)
            println()
            println(string("Number of consecutive failed iterations = ", tfail))
            println()
            min(varn/2, 0.005)
        else
            varn
        end

        println(string("Updated perturbation variance is ", varn))

        realxnew = map((i, x) -> map_range_inv(x, opt_params[i][1], opt_params[i][2]), h, eachindex(h))

        #update Fs, Xs, parameter vectors, and outputs 
        push!(errs, errnew)
        push!(xs, realxnew)
        push!(params,  paramsnew)
        push!(outputs, outputnew)
    end
    (errs, [a[h] for a in params], outputs, xs, resultsdict)
end

function runHORDopt_trials(optfunc::Function, opt_params, nmax, isp = []; resultsdict = (), pnames = ["Parameter $n" for n in 1:length(opt_params)], pconvert = map(identity, opt_params))
    (errs, params, outputs, xs, resultsdict) = run_HORDopt(optfunc, opt_params, 1, nmax, isp, pnames = pnames, pconvert = pconvert, resultsdict = resultsdict)
    h = findall(a -> length(a) == 2, opt_params)
    ih = setdiff(eachindex(opt_params), h)
    (newerr, bestind) = findmin(errs)
    besterr = Inf
    results = [(1, errs[bestind], params[bestind], outputs[bestind])]
    id = 2
    while newerr < besterr
        println("=================================================================")
        println("=====================Starting Trial $id==========================")
        println("=================================================================")
        besterr = newerr
        (errs, params, outputs, xs, resultsdict) = run_HORDopt(optfunc, opt_params, id, nmax, resultsdict = resultsdict, pnames = pnames, pconvert = pconvert)
        (newerr, bestind) = findmin(errs)
        push!(results, (id, errs[bestind], params[bestind], outputs[bestind]))
        id += 1
    end

    fixednames = pnames[ih]
    fixedparams = iterate(keys(resultsdict))[1][ih]
    return results, resultsdict, pnames[h], fixednames, fixedparams
end