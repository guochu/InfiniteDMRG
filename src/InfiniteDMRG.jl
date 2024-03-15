module InfiniteDMRG


# infinite MPS
export AbstractInfiniteMPS, InfiniteMPS, prodimps, randomimps, unitcell_size, InfiniteOrthogonalize

# infinite MPO
export InfiniteMPO, randomimpo

# environments
export left_boundary, right_boundary, leading_eigenvalue

# algorithm
export num_period
export iterative_compress!, iterative_compress


using Logging: @warn
using Reexport, KrylovKit
@reexport using SphericalTensors, DMRG
using SphericalTensors: QR, LQ, SVD
const TK = SphericalTensors
using LinearAlgebra: eigen, Hermitian
using DMRG: left_embedders, right_embedders
using DMRG: updateright, updateleft, OverlapTransferMatrix, MatrixProductOrthogonalAlgorithm

#default settings
module Defaults
	const maxiter = 10000 # for Arnoldi iteration to find largest eigenpair
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
include("states/mixedcanonical.jl")
include("states/linalg.jl")
include("states/orthutil.jl")
include("states/orth.jl")
include("states/orth2.jl")
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
include("algorithms/approximate.jl")

end