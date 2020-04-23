function phi(x1, x2)
	#euclidean distance between x1 and x2 cubed
	n = norm(x1 .- x2, 2)^3
    # if isinf(n) || isnan(n)
    #     println("phi value is $n which is illegal, replacing with $(maxintfloat(typeof(n)))")
    #     println("X1 = $x1")
    #     println("X2 = $x2")

        #discovered error occured when x vectors themselves contained Inf values
        #doing this replacement doesn't fix the problem with the pseudoinverse, also on retrials
        #there were repeated points with Inf values which caused NaN values.  So this replacement
        #is only useful for debugging but not for solving the problem.
    #     return maxintfloat(typeof(n))
    # else
    #     return n
    # end
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

function refineparams(optparams, bestparams, pconvert, pconvertinv)
#narrow range for parameter search based on the last
#observed best parameters
    optind = findall(a -> length(a)>1, optparams)
    Tuple([begin
        if length(p) == 1
            p
        else
            newp1 = pconvertinv[i](pconvert[i]((pconvertinv[i](bestparams[i]) + p[1])/2))
            newp2 = pconvertinv[i](pconvert[i]((pconvertinv[i](bestparams[i]) + p[2])/2))
            T = typeof(newp1)
            if T <: Integer
                if newp1 == newp2
                    (newp1-1, newp2+1)
                else
                    (newp1, newp2)
                end
            else
                if abs(newp1/newp2 - 1) < 0.01
                    m = (newp1+newp2)/2
                    d = m*0.005
                    (T(m-d), T(m+d))
                else   
                    (newp1, newp2)
                end
            end            
        end
    end
    for (i, p) in enumerate(optparams)])
end

function centerparams(optparams, bestparams, pconvert, pconvertinv, scale = 1.0)
#narrow range for parameter search based on the last
#observed best parameters
    optind = findall(a -> length(a)>1, optparams)
    Tuple([begin
        if length(p) == 1
            p
        else
            # d = scale*(p[2] - p[1])
            pc = pconvertinv[i](bestparams[i])
            d = scale*pc

            if d == 0
                d1 = p[2] - p[1]
                newp1 = pconvertinv[i](pconvert[i])(pc)
                newp2 = newp1 + d1*scale
            else
                newp1 = pconvertinv[i](pconvert[i](pc - d/2))
                newp2 = pconvertinv[i](pconvert[i](pc + d/2))
            end
            T = typeof(newp1)
            if T <: Integer
                if newp1 == newp2
                    (newp1-1, newp2+1)
                else
                    (newp1, newp2)
                end
            else
                if abs(newp1/newp2 - 1) < 0.01
                    m = (newp1+newp2)/2
                    d = m*0.005
                    (T(m-d), T(m+d))
                else   
                    (newp1, newp2)
                end
            end            
        end
    end
    for (i, p) in enumerate(optparams)])
end