function phi(x1, x2)
	#euclidean distance between x1 and x2 cubed
	norm(x1 .- x2, 2)^3
end

#calculate phi matrix using an input of previous hyperparamter vectors
function calc_phi_mat(Xs)
    n = length(Xs)
    mapreduce(vcat, 1:n) do i
        map(j -> phi(Xs[i], Xs[j]), 1:n)'
    end            
end

function formP(Xs)
    n = length(Xs)
    mapreduce(vcat, 1:n) do i
        [Xs[i]' 1.0]
    end
end

function surrogate_model(x, Xs, lambda, b, a)
#surrogate model output for hyperparameter vector input x
#requires knowing current list of n hyperparameter vectors and other
#model parameters lambda (vector of n constants), b (vector of D constants), 
#and a (constant) with n hyperparameter vector inputs
    n = length(Xs)
    D = length(Xs[1])
    mapreduce(+, 1:n) do i
        lambda[i]*phi(x, Xs[i]) + dot(b, x) + a
    end
end

function calc_phiN(D, n, n0, Nmax)
    phi0 = min(20/D, 1)
    phi0*(1 - log(n - n0+1)/log(Nmax - n0))
end

function wrapX(x)
#wraps a value x into the range 0, 1 treating it as periodic
#boundry conditions so 1.2 => 0.2, -2.4 => 0.6
    if x > 1
        x - floor(Int64, x)
    elseif x < 0
        1.0 + x - ceil(Int64, x)
    else
        x
    end
end


function fill_omegaN(xbest, phiN, varN)
    D = length(xbest)
    m = 100*D

    #generate m new candidate points
    Ys = map(1:m) do _
        #select coordinates to perturb with probability phiN and perterb sampled from a normal
        #distribution with 0 mean and variance of varN
        p = randn(D)*sqrt(varN).*(rand(D) .<= phiN)

        #ensure new x values are above 0
        # xnew = [max(0, x) for x in (xbest .+ p)]
        xnew = xbest .+ p #tryign to allow x to extend beyond 0, 1 range with pconvert function keeping values valid
        
        # wrapX.(xnew) #restrict x to range of initial points otherwise comment this out
    end
end

function maprange(x, ymin, ymax)
#map an input value from 0 to 1 to an output range from ymin to ymax
#using linear scaling
    @assert (ymax > ymin) string("the range ", ymin, " to ", ymax, " is not valid")
    d = ymax - ymin
    y = x*d + ymin
end

function map_range_inv(x, ymin, ymax)
#map an input value from ymin to ymax to an output range of 0 to 1
#using linear scaling
    @assert (ymax > ymin) string("the range ", ymin, " to ", ymax, " is not valid")
    d = ymax - ymin
    y = (x - ymin)*1/d
end

function permuteweight(w)
#cyclically permute weight values
    if w == 0.3
        0.5
    elseif w == 0.5
        0.8
    elseif w == 0.8
        0.95
    elseif w == 0.95
        0.3
    end 
end