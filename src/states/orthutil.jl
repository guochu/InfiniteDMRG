struct InfiniteOrthogonalize{T<:TruncationScheme} <: MatrixProductOrthogonalAlgorithm
	trunc::T
	normalize::Bool
	toleig::Float64 
	maxitereig::Int 
end
InfiniteOrthogonalize(; trunc::TruncationScheme=DefaultTruncation, normalize::Bool=true, toleig::Real=Defaults.tolgauge, maxitereig::Int=Defaults.maxiter) = InfiniteOrthogonalize(
	trunc, normalize, convert(Float64, toleig), maxitereig)
InfiniteOrthogonalize(alg::Orthogonalize; toleig::Real=Defaults.tolgauge, maxitereig::Int=Defaults.maxiter) = InfiniteOrthogonalize(
						trunc=alg.trunc, normalize=alg.normalize, toleig=toleig, maxitereig=maxitereig)

function leading_boundaries(tn::InfiniteMPS; kwargs...) 
	Adata = [tn[i] for i in 1:unitcell_size(tn)]
	cell = TransferMatrix(Adata, Adata)
	# if dim(space_l(tn)) >= 20
	vl, vr = random_boundaries(cell)
	vl = vl * vl'
	vr = vr * vr'
	if bond_dimension(tn) > 4*length(cell)
		left_eigenvalue, left_eigenvector = largest_eigenpair(x -> x * cell, vl; kwargs...)
		right_eigenvalue, right_eigenvector = largest_eigenpair(x -> cell * x, vr; kwargs...)		
	else
		dcell = convert(TensorMap, cell)
		left_eigenvalue, left_eigenvector = largest_eigenpair(x -> x * dcell, permute(vl, (), (1,2)); kwargs...)
		left_eigenvector = permute(left_eigenvector, (1,), (2,))
		right_eigenvalue, right_eigenvector = largest_eigenpair(x -> dcell * x, permute(vr, (2,1), ()); kwargs...)
		right_eigenvector = permute(right_eigenvector, (2,), (1,))
	end

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
	return left_eigenvalue, normalize_trace!(left_eigenvector), normalize_trace!(right_eigenvector)
end

const EIGENVALUE_IMAG_TOL = 1.0e-12
const EIGENVECTOR_IMAG_TOL = 1.0e-6


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

# function chol_split(m::AbstractMatrix{<:Number})
#     evals, evecs = eigen(Hermitian(m))
#     z = zero(eltype(evals))
#     sqrt_evals = zeros(eltype(evals), length(evals))
#     for i in 1:length(evals)
#     	if evals[i] > z
#     		sqrt_evals[i] = sqrt(evals[i])
#     	end
#     end
#     return Diagonal(sqrt_evals) * evecs'
# end

# function chol_split(m::MPSBondTensor) 
#     r = similar(m)
#     for (c, b) in blocks(m)
#     	b2 = chol_split(b)
#     	copy!(block(r, c), b2)
#     end
#     return r
# end

"""
    fixedpoint(A, x₀, which::Symbol, alg) -> val, vec

Compute the fixedpoint of a linear operator `A` using the specified eigensolver `alg`. The
fixedpoint is assumed to be unique.
"""
function fixedpoint(A, x₀, which::Symbol, alg::Lanczos)
    vals, vecs, info = eigsolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warn "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end

    return vals[1], vecs[1]
end

function fixedpoint(A, x₀, which::Symbol, alg::Arnoldi)
    TT, vecs, vals, info = schursolve(A, x₀, 1, which, alg)

    if info.converged == 0
        @warn "fixedpoint not converged after $(info.numiter) iterations: normres = $(info.normres[1])"
    end
    if size(TT, 2) > 1 && TT[2, 1] != 0
        @warn "non-unique fixedpoint detected"
    end

    return vals[1], vecs[1]
end

