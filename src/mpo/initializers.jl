
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

"""
	InfiniteMPO(h::MPOHamiltonian, L::Int) 
	
Conversion of an MPOHamiltonian into a standard MPO
"""
function InfiniteMPO(h::MPOHamiltonian) 
	L = length(h)
	(L >= 1) || throw(ArgumentError("size of MPO must at least be 1"))
	(h[1].leftspaces == h[end].rightspaces) || throw(SpaceMismatch("boundary space mismatch"))
	T = scalartype(h)
	S = spacetype(h)

	mpotensors = Vector{mpotensortype(S, T)}(undef, L)
	embedders = PeriodicArray([right_embedders(T, h[i].rightspaces...) for i in 1:length(h)])

	for n in 1:L
		tmp = TensorMap(zeros, T, space(embedders[n-1][1], 2)' * h[n].pspace ← space(embedders[n][1], 2)' * h[n].pspace )
		for (i, j) in DMRG.opkeys(h[n])
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * h[n, i, j][1,-2,2,-4] * embedders[n][j][2, -3]
		end
		for (i, j) in DMRG.scalkeys(h[n])
			# iden = h[n].Os[i, j] * isomorphism(Matrix{T}, h[n].pspace, h[n].pspace)
			# @tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * embedders[n][j][1, -3] * iden[-2, -4] 
			@tensor tmp[-1, -2, -3, -4] += conj(embedders[n-1][i][1, -1]) * h[n, i, j][1,-2,2,-4] * embedders[n][j][2, -3]
		end
		mpotensors[n] = tmp
	end
	return InfiniteMPO(mpotensors)
end

function InfiniteMPO(f, ::Type{T}, physpaces::Vector{S}, virtualpaces::Vector{S}) where {T <: Number, S <: ElementarySpace}
	@assert length(physpaces) == length(virtualpaces)
	L = length(physpaces)
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	virtualpaces2 = PeriodicArray(virtualpaces)
	data = [TensorMap(f, T, virtualpaces2[i] ⊗ physpaces[i] ← virtualpaces2[i+1] ⊗ physpaces[i] ) for i in 1:L]
	return InfiniteMPO(data)
end

function InfiniteMPO(f, ::Type{T}, physpaces::Vector{S}, maxvirtualspace::S; right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = right
	for i in 2:L
		virtualpaces[i] = infimum(fuse(virtualpaces[i-1], physpaces[i-1], physpaces[i-1]'), maxvirtualspace)
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1], physpaces[i]))
	end
	return InfiniteMPO(f, T, physpaces, virtualpaces[1:L])
end

"""
	randomimpo(::Type{T}, physpaces::Vector{S}; D::Int, left::S, right::S)

Return a random MPO.

Each virtual space has multiplicity 1 to allow the largest number of different 
quantum numbers to be explored under bond dimension D
"""
function randomimpo(::Type{T}, physpaces::Vector{S}; D::Int, right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	virtualpaces = DMRG._max_mpo_virtual_spaces(physpaces, D, right, right)
	return InfiniteMPO(randn, T, physpaces, virtualpaces[1:length(physpaces)])
end
randomimpo(physpaces::Vector{S}; kwargs...) where {S <: ElementarySpace} = randomimpo(Float64, physpaces; kwargs...)

