module HORDOpt

using LinearAlgebra
using Random

include("parameter_utils.jl")
include("hyperparameter_search.jl")

export run_HORDopt, convert_params

end # module
