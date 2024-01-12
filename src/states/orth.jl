function TransferMatrix(a::InfiniteMPS, b::InfiniteMPS)
	Adata, Bdata = get_common_data(a, b)
	return TransferMatrix(Adata, Bdata)
end
TransferMatrix(a::InfiniteMPS) = TransferMatrix(a, a)

function leading_boundaries(tn::InfiniteMPS) 
	# Adata = [tn[i] for i in 1:unitcell_size(tn)]
	cell = TransferMatrix(tn)
	# if dim(space_l(tn)) >= 20
	vl, vr = random_boundaries(cell)
	left_eigenvalue, left_eigenvector = _eigsolve_pos(x -> transfer_left(x, cell), vl * vl')
	right_eigenvalue, right_eigenvector = _eigsolve_pos(x -> transfer_right(x, cell), vr * vr')
	# else
	# 	m = convert(TensorMap, cell)
	# 	left_eigenvalues, left_eigenvectors = eig!(permute(m, (3,4), (1,2)))
	# 	println(left_eigenvalues)
	# 	right_eigenvalues, right_eigenvectors = eig!(m)
	# 	println(right_eigenvalues)
	# 	left_eigenvalue = left_eigenvalues[1]
	# 	right_eigenvalue = right_eigenvalues[1]
	# 	left_eigenvector = permute(left_eigenvectors[1], (1,), (2,))
	# 	right_eigenvector = permute(right_eigenvectors[1], (2,), (1,))
	# end
	(left_eigenvalue ≈ right_eigenvalue) || @warn "left and right dominate eigenvalues $(left_eigenvalue) and $(right_eigenvalue) mismatch"
	return left_eigenvalue, left_eigenvector, right_eigenvector
end

function _eigsolve(f, v0)
	eigenvalues, eigenvectors, info = eigsolve(f, v0, 1, :LM, Arnoldi(krylovdim=30, maxiter=Defaults.maxiter))
	(info.converged >= 1) || error("dominate eigendecomposition fails to converge")
	eigenvalue = eigenvalues[1]
	eigenvector = eigenvectors[1]
	return eigenvalue, eigenvector
end

function _eigsolve_pos(f, v0)
	eigenvalue, eigenvector = _eigsolve(f, v0)
	eigenvector = normalize_trace!(eigenvector)
	if (scalartype(v0) <: Real) && (scalartype(eigenvector) <: Complex)
		(norm(imag(eigenvector)) < 1.0e-8 ) || @warn "norm of imaginary part of eigenvector is $(norm(imag(eigenvector)))"
		(abs(imag(eigenvalue)) < 1.0e-12) || @warn "imaginary part of eigenvalue is $(imag(eigenvalue))"
		return real(eigenvalue), real(eigenvector)
	end
	return eigenvalue, eigenvector
end

const CHOL_SPLIT_TOL = 1.0e-12

function approximate!(y::InfiniteMPS, x::InfiniteMPS)
	@assert unitcell_size(y) == unitcell_size(x)
end

"""
	canonicalize!(x::InfiniteMPS; alg::Orthogonalize)
Preparing an infinite symmetric mps into (right-)canonical form
Reference: PHYSICAL REVIEW B 78, 155117 
"""
function DMRG.canonicalize!(x::InfiniteMPS; alg::Orthogonalize{TK.SVD, TruncationDimCutoff} = Orthogonalize(TK.SVD(), DefaultTruncation, normalize=false))
	# alg.normalize || throw(ArgumentError("normalization has been doen for infinite mps"))
	eta, Vl, Vr = leading_boundaries(x)
	# println("eta is ", eta)
	tolchol = alg.trunc.ϵ / 10
	Y = chol_split(Vl, tolchol)
	X = chol_split(Vr, tolchol)'
	U, S, V = tsvd!(Y * X, trunc=alg.trunc)
	alg.normalize && normalize!(S)
	
	m = (S * V) / X
	L = unitcell_size(x)
	for i in 1:(L-1)
		@tensor xj[1,3; 4] := m[1, 2] * x[i][2,3,4]
		x[i], m = leftorth!(xj, alg=TK.QRpos())
	end
	m2 = Y \ (U * S)
	x[L] = @tensor tmp[1,3; 5] := m[1,2] * x[L][2,3,4] * m2[4,5]
	
	for i in L:-1:2
		# @tensor xj[1; 2 4] := x[i][1,2,3] * m[3,4]
		v, ss, xj2, err = tsvd(x[i], (1,), (2,3), trunc=alg.trunc)
		x[i] = permute(xj2, (1,2), (3,))
		alg.normalize && (normalize!(ss))
		x.s[i] = ss
		m = v * ss
		x[i-1] = @tensor tmp[1,2;4] := x[i-1][1,2,3] * m[3,4]
	end
	x[1] = @tensor tmp[1,3; 4] := inv(S)[1,2] * x[1][2,3,4] 
	x.s[1] = S
	return x
end

function normalize_trace!(x::TensorMap)
	return lmul!(1 / tr(x), x)
end


function chol_split(m::AbstractMatrix{<:Number}, tol::Real)
    # println("m is hermitian? $(maximum(abs.(m - m'))).")
    evals, evecs = eigen(Hermitian(m))
    # println("eigenvalues ", evals)
    k = length(evals)+1
    for i in 1:length(evals)
    	if evals[i] > tol
    		k = i
    		break
    	end
    end
    # positivity check
    (maximum(abs, view(evals, 1:k-1), init=0.) < CHOL_SPLIT_TOL) || @warn "input matrix is not positive (with eigenvalue $(maximum(abs, view(evals, 1:k-1), init=0.))"
    return Diagonal(sqrt.(evals[k:end])) * evecs[:, k:end]'
end

function chol_split(m::MPSBondTensor{S}, tol::Real) where {S}
    r = empty(m.data)
    dims = TK.SectorDict{sectortype(S), Int}()
    for (c, b) in blocks(m)
    	b2 = chol_split(b, tol)
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
