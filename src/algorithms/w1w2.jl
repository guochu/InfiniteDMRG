
@with_kw struct InfiniteWI <: TimeEvoMPOAlgorithm
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

@with_kw struct InfiniteWII <: TimeEvoMPOAlgorithm
	tol::Float64 = Defaults.tol
	maxiter::Int = Defaults.maxiter
end

DMRG.timeevompo(m::SchurMPOTensor, dt::Number, alg::InfiniteWI) = timeevompo(m, dt, WI(tol=alg.tol, maxiter=alg.maxiter))
DMRG.timeevompo(m::SchurMPOTensor, dt::Number, alg::InfiniteWII) = timeevompo(m, dt, WII(tol=alg.tol, maxiter=alg.maxiter))
DMRG.timeevompo(m::MPOHamiltonian{<:SchurMPOTensor}, dt::Number, alg::Union{InfiniteWI, InfiniteWII}) = MPOHamiltonian([timeevompo(mj, dt, alg) for mj in m.data])
