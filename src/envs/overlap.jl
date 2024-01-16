

"""
	struct InfiniteOverlapCache{_A, _B, _C}

A is the bra, B is ket, ⟨A|B⟩, in iterative algorithms,
A is the output, B is the input
"""
struct InfiniteOverlapCache{_A, _B, _L, _R, _T} <: AbstractInfiniteCache
	A::_A
	B::_B
	left::_L
	right::_R
	η::_T
end

Base.length(x::InfiniteOverlapCache) = max(length(x.A), length(x.B))
bra(x::InfiniteOverlapCache) = x.A
ket(x::InfiniteOverlapCache) = x.B
left_boundary(x::InfiniteOverlapCache) = x.left
right_boundary(x::InfiniteOverlapCache) = x.right
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
	return InfiniteOverlapCache(psiA, psiB, lmul!(1/trace, left_eigenvector), right_eigenvector, left_eigenvalue)
end

function DMRG.leftenv(x::InfiniteOverlapCache, i::Int)
	pos = r_start(i, length(x))
	Adata, Bdata = get_common_data(x.A, x.B, pos, i-1)
	return left_boundary(x) * TransferMatrix(Adata, Bdata)
end
function DMRG.rightenv(x::InfiniteOverlapCache, i::Int)
	pos = r_stop(i, length(x))
	Adata, Bdata = get_common_data(x.A, x.B, i+1, pos)
	return TransferMatrix(Adata, Bdata) * right_boundary(x)
end
DMRG.value(x::InfiniteOverlapCache, nperiod::Int=1) = leading_eigenvalue(x)^nperiod