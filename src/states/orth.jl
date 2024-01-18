function leading_boundaries(tn::InfiniteMPS; kwargs...) 
	Adata = [tn[i] for i in 1:unitcell_size(tn)]
	cell = TransferMatrix(Adata, Adata)
	# if dim(space_l(tn)) >= 20
	vl, vr = random_boundaries(cell)
	left_eigenvalue, left_eigenvector = largest_eigenpair_pos(x -> x * cell, vl * vl'; kwargs...)
	right_eigenvalue, right_eigenvector = largest_eigenpair_pos(x -> cell * x, vr * vr'; kwargs...)
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

const EIGENVALUE_IMAG_TOL = 1.0e-12
const EIGENVECTOR_IMAG_TOL = 1.0e-6


function largest_eigenpair_pos(f, v0; kwargs...)
	eigenvalue, eigenvector = largest_eigenpair(f, v0; kwargs...)
	return eigenvalue, normalize_trace!(eigenvector)
end

function largest_eigenpair(f, v0; kwargs...)
	eigenvalue, eigenvector = _eigsolve_bare(f, v0; kwargs...)
	eigenvector = normalize_angle!(eigenvector)
	if (scalartype(v0) <: Real) && (scalartype(eigenvector) <: Complex)
		if (norm(imag(eigenvector)) / norm(real(eigenvector)) < EIGENVECTOR_IMAG_TOL) && (abs(imag(eigenvalue)) < EIGENVALUE_IMAG_TOL)
			return real(eigenvalue), real(eigenvector)
		end
	end
	return eigenvalue, eigenvector	
end

function _eigsolve_bare(f, v0; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
	eigenvalues, eigenvectors, info = eigsolve(f, v0, 1, :LM, Arnoldi(;krylovdim=30, maxiter=maxiter, tol=tol))
	(info.converged >= 1) || error("dominate eigendecomposition fails to converge")
	eigenvalue = eigenvalues[1]
	eigenvector = eigenvectors[1]
	return eigenvalue, eigenvector
end


@with_kw struct InfiniteOrthogonalize <: MatrixProductOrthogonalAlgorithm
	trunc::TruncationDimCutoff = DefaultTruncation
	normalize::Bool=true
	toleig::Float64 = Defaults.tolgauge
	maxitereig::Int = Defaults.maxiter
end

DMRG.canonicalize!(x::InfiniteMPS; alg::MatrixProductOrthogonalAlgorithm=InfiniteOrthogonalize(normalize=false)) = canonicalize!(x, alg)
DMRG.canonicalize!(x::InfiniteMPS, alg::Orthogonalize{TK.SVD, TruncationDimCutoff}) = canonicalize!(
	x, InfiniteOrthogonalize(trunc=alg.trunc, normalize=alg.normalize, toleig=Defaults.tolgauge, maxitereig=Defaults.maxiter))
"""
	canonicalize!(x::InfiniteMPS; alg::Orthogonalize)
Preparing an infinite symmetric mps into (right-)canonical form
Reference: PHYSICAL REVIEW B 78, 155117 
"""
function DMRG.canonicalize!(x::InfiniteMPS, alg::InfiniteOrthogonalize)
	# alg.normalize || throw(ArgumentError("normalization has been doen for infinite mps"))
	# println("eta is ", eta)
	tolchol = max(alg.trunc.ϵ * alg.trunc.ϵ, alg.toleig)
	eta, Vl, Vr = leading_boundaries(x, tol=tolchol/10, maxiter=alg.maxitereig)

	Y = chol_split(Vl, tolchol)
	X = chol_split(Vr, tolchol)'
	U, S, V = stable_tsvd!(Y * X, trunc=alg.trunc)
	alg.normalize && normalize!(S)
	
	# m = (S * V) / X
	m = U' * Y
	L = unitcell_size(x)
	for i in 1:(L-1)
		@tensor xj[1,3; 4] := m[1, 2] * x[i][2,3,4]
		x[i], m = leftorth!(xj, alg=TK.QRpos())
	end
	# m2 = Y \ (U * S)
	m2 = X * V'
	x[L] = @tensor tmp[1,3; 5] := m[1,2] * x[L][2,3,4] * m2[4,5]
	
	for i in L:-1:2
		# @tensor xj[1; 2 4] := x[i][1,2,3] * m[3,4]
		v, ss, xj2, err = stable_tsvd(x[i], (1,), (2,3), trunc=alg.trunc)
		x[i] = permute(xj2, (1,2), (3,))
		alg.normalize && (normalize!(ss))
		x.s[i] = ss
		m = v * ss
		x[i-1] = @tensor tmp[1,2;4] := x[i-1][1,2,3] * m[3,4]
	end
	# @tensor trace = Vl[1,2] * Vr[2,1]
	# println("trace is ", trace)
	# x[1] = @tensor tmp[1,3; 4] := (1/sqrt(trace)) * inv(S)[1,2] * x[1][2,3,4] 
	# x.s[1] = alg.normalize ? S : lmul!(1 / sqrt(trace), S) 

	# x[1] = @tensor tmp[1,3; 4] := inv(S)[1,2] * x[1][2,3,4] 
	# x[1] = permute(S \ permute(x[1], (1,), (2,3)), (1,2), (3,))
	# x.s[1] = S

	# Is it better to truncate the first site again?
	v, ss, xj2, err = stable_tsvd(x[1], (1,), (2,3), trunc=alg.trunc)
	if alg.normalize
		normalize!(ss)
		x[1] = permute(xj2, (1,2), (3,))
		x.s[1] = ss
	else
		n = norm(ss) / norm(S)
		x[1] = lmul!(n, permute(xj2, (1,2), (3,)))
		x.s[1] = lmul!(1/n, ss) 
	end
	x[end] = x[end] * v
	return x
end

function normalize_trace!(x::TensorMap)
	return lmul!(1 / tr(x), x)
end
function normalize_angle!(x::TensorMap)
	v = argmax(abs, argmax(abs, y[2]) for y in blocks(x) ) 
	return lmul!(conj(TK._safesign(v)), x)
end

const CHOL_SPLIT_TOL = 1.0e-12

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
    (maximum(-, view(evals, 1:k-1), init=0.) < CHOL_SPLIT_TOL) || @warn "input matrix is not positive (with negative eigenvalue $(argmax(-, view(evals, 1:k-1)))"
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
