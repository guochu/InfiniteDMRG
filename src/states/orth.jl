function leading_boundaries(x::M, y::M) where {M <: AbstractInfiniteTN}
	Adata, Bdata = get_common_data(x, y)
	cell = TransferMatrix(Adata, Bdata)
	vl, vr = random_boundaries(cell)
	left_eigenvalue, left_eigenvector, info = eigsolve(x -> transfer_left(x, cell), vl, 1, :LM, Arnoldi())
	(info.converged >= 1) || error("left dominate eigendecomposition fails to converge")
	right_eigenvalue, right_eigenvector, info = eigsolve(x -> transfer_right(x, cell), vr, 1, :LM, Arnoldi())
	(info.converged >= 1) || error("right dominate eigendecomposition fails to converge")
	(left_eigenvalue[1] ≈ right_eigenvalue[1]) || throw(ArgumentError("left and right dominate eigenvalue mismatch"))
	return left_eigenvalue[1], normalize_trace!(left_eigenvector[1]), normalize_trace!(right_eigenvector[1])
end


"""
	canonicalize!(x::InfiniteMPS; alg::Orthogonalize)
Preparing an infinite symmetric mps into (right-)canonical form
Reference: PHYSICAL REVIEW B 78, 155117 
"""
function DMRG.canonicalize!(x::InfiniteMPS; alg::Orthogonalize = Orthogonalize(TK.SVD(), DefaultTruncation, normalize=true))
	eta, Vl, Vr = leading_boundaries(x, x)
	# println("eta is ", eta)
	Y = chol_split(Vl)
	X = chol_split(Vr)'
	U, S, V = tsvd!(Y * x.s[1] * X)

	alg.normalize && normalize!(S)
	m = (S * V) / X
	L = unitcell_size(x)
	for i in 1:(L-1)
		@tensor xj[1,3; 4] := m[1, 2] * x[i][2,3,4]
		x[i], m = leftorth!(xj, alg=TK.QR())
	end
	x[L] = @tensor tmp[1,3; 5] := m[1, 2] * x[L][2,3,4] * inv(x.s[L+1])[4,5]
	m = Y \ (U * S)
	for i in L:-1:2
		@tensor xj[1; 2 4] := x[i][1,2,3] * m[3,4]
		v, ss, xj2, err = tsvd!(xj, trunc=alg.trunc)
		x[i] = permute(xj2, (1,2), (3,))
		alg.normalize && (normalize!(ss))
		x.s[i] = ss
		m = v * ss
	end
	x[1] = @tensor tmp[1,3; 5] := inv(S)[1,2] * x[1][2,3,4] * m[4,5]
	x.s[1] = S
	return x
end

function normalize_trace!(x::TensorMap)
	return lmul!(1 / tr(x), x)
end

const CHOL_SPLIT_TOL = 1.0e-14

function chol_split(m::AbstractMatrix{<:Number})
    # println("m is hermitian? $(maximum(abs.(m - m'))).")
    evals, evecs = eigen(Hermitian(m))
    # println("eigenvalues ", evals)
    k = length(evals)+1
    for i in 1:length(evals)
    	if evals[i] > CHOL_SPLIT_TOL
    		k = i
    		break
    	end
    	# positive check
        # println("$(evals[i])--------------------")
        (abs(evals[i]) < CHOL_SPLIT_TOL) || error("input matrix is not positive")
    end
    return Diagonal(sqrt.(evals[k:end])) * evecs[:, k:end]'
end

function chol_split(m::MPSBondTensor{S}) where {S}
    r = empty(m.data)
    dims = TK.SectorDict{sectortype(S), Int}()
    for (c, b) in blocks(m)
    	b2 = chol_split(b)
    	if !isempty(b2)
    		# println(c, " ", size(b2))
    		r[c] = b2
    		dims[c] = size(b2, 1)
    	end
    end
    W = S(dims)
    return TensorMap(r, W, domain(m))
end

# function Base.inv(t::DiagonalMap)
#     cod = codomain(t)
#     dom = domain(t)
#     for c in union(blocksectors(cod), blocksectors(dom))
#         blockdim(cod, c) == blockdim(dom, c) ||
#             throw(SpaceMismatch("codomain $cod and domain $dom are not isomorphic: no inverse"))
#     end

#     data = empty(t.data)
#     for (c, b) in blocks(t)
#         data[c] = inv(b)
#     end
#     return DiagonalMap(data, domain(t)←codomain(t))
# end
