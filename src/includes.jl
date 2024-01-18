using Logging: @warn
using Reexport, KrylovKit, Parameters
@reexport using SphericalTensors, DMRG
const TK = SphericalTensors
using LinearAlgebra: eigen, Hermitian
using DMRG: left_embedders, right_embedders
using DMRG: updateright, updateleft, OverlapTransferMatrix, MatrixProductOrthogonalAlgorithm

#default settings
module Defaults
	const maxiter = 500 # for Arnoldi iteration to find largest eigenpair
	const D = 100 # default bond dimension 
	const tolgauge = 1e-14 # for MPS truncation
	const tol = 1e-12 # for DMRG iteration
	const tollanczos = 1.0e-10 # for lanczos eigensolver
	const tolexp = 1.0e-8 # for local eigen in DMRG
	const verbosity = 1
end

const DefaultTruncation = TruncationDimCutoff(D=Defaults.D, ϵ=1.0e-12, add_back=0)

# transfer matrix
include("transfer.jl")

# infinite MPS
include("states/abstractmps.jl")
include("states/bondview.jl")
include("states/infinitemps.jl")
include("states/linalg.jl")
include("states/orth.jl")
include("states/initializers.jl")


# infinite mpo
include("mpo/infinitempo.jl")
include("mpo/linalg.jl")
include("mpo/initializers.jl")

# environments
include("envs/environments.jl")

# algorithms
include("algorithms/expecs.jl")
# include("algorithms/tdvp.jl")
# include("algorithms/approximate.jl")