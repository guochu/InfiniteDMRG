

"""
	struct InfiniteOverlapCache{_M, _L, _R, _T}

A is the bra, B is ket, ⟨A|B⟩, in iterative algorithms,
A is the output, B is the input
"""
struct InfiniteOverlapCache{_M, _L, _R, _T} <: AbstractInfiniteCache
	A::_M
	B::_M
	left::_L
	right::_R
	η::_T
end

Base.length(x::InfiniteOverlapCache) = max(length(x.A), length(x.B))
bra(x::InfiniteOverlapCache) = x.A
ket(x::InfiniteOverlapCache) = x.B
left_boundary(x::InfiniteOverlapCache) = x.left[1]
right_boundary(x::InfiniteOverlapCache) = x.right[end]
leading_eigenvalue(x::InfiniteOverlapCache) = x.η
unitcell_size(x::InfiniteOverlapCache) = length(x)

function DMRG.environments(psiA::M, psiB::M) where {M <: Union{InfiniteMPS, InfiniteMPO}}
	Adata, Bdata = get_common_data(psiA, psiB)
	cell = TransferMatrix(Adata, Bdata)
	vl, vr = random_boundaries(cell)
	left_eigenvalue, left_eigenvector = largest_eigenpair(x -> x * cell, vl)
	right_eigenvalue, right_eigenvector = largest_eigenpair(x -> cell * x, vr)
	(left_eigenvalue ≈ right_eigenvalue) || @warn "left and right dominate eigenvalues $(left_eigenvalue) and $(right_eigenvalue) mismatch"
	@tensor trace = left_eigenvector[1,2] * right_eigenvector[2,1]

	L = length(Adata)
	left = Vector{typeof(left_eigenvector)}(undef, L)
	left[1] = lmul!(1/trace, left_eigenvector)
	for i in 2:L
		Adata_i, Bdata_i = get_common_data(psiA, psiB, i-1, i-1)
		left[i] = left[i-1] * TransferMatrix(Adata_i, Bdata_i)
	end
	right = Vector{typeof(right_eigenvector)}(undef, L)
	right[end] = right_eigenvector
	for i in L-1:-1:1
		Adata_i, Bdata_i = get_common_data(psiA, psiB, i+1, i+1)
		right[i] = TransferMatrix(Adata_i, Bdata_i) * right[i+1]
	end
	return InfiniteOverlapCache(psiA, psiB, PeriodicArray(left), PeriodicArray(right), left_eigenvalue)
end

DMRG.leftenv(x::InfiniteOverlapCache, i::Int) = x.left[i]
DMRG.rightenv(x::InfiniteOverlapCache, i::Int) = x.right[i]

function setleftenv!(x::InfiniteOverlapCache, i::Int, v) 
	x.left[i] = v
end
function setrightenv!(x::InfiniteOverlapCache, i::Int, v)
	x.right[i] = v
end

DMRG.value(x::InfiniteOverlapCache, nperiod::Int=1) = leading_eigenvalue(x)^nperiod
