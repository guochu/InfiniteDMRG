

"""
    Base.:+(psiA::InfiniteMPS, psiB::InfiniteMPS) 
Addition of two MPSs
"""
function Base.:+(psiA::InfiniteMPS, psiB::InfiniteMPS) 
	Adata, Bdata = get_common_data(psiA, psiB)
	cellsize = length(Adata)
    T = promote_type(scalartype(psiA), scalartype(psiB))
    embedders = PeriodicArray([right_embedders(T, space_r(aj)', space_r(bj)') for (aj, bj) in zip(Adata, Bdata)])
    A = mpstensortype(spacetype(psiA), T)
    r = Vector{A}(undef, cellsize)
    for i in 1:cellsize
        @tensor m1[-1 -2; -3] := (embedders[i-1][1])'[-1, 1] * psiA[i][1,-2,2] * embedders[i][1][2, -3]
        @tensor m1[-1 -2; -3] += (embedders[i-1][2])'[-1, 1] * psiB[i][1,-2,2] * embedders[i][2][2, -3]
        r[i] = m1
    end
    return InfiniteMPS(r)
end


get_common_data(a, b) = get_common_data(a, b, 1, max(unitcell_size(a), unitcell_size(b)))
function get_common_data(a, b, start::Int, _end::Int)
    Adata = [a[i] for i in start:_end]
    Bdata = [b[i] for i in start:_end]
    return Adata, Bdata
end

function TK.dot(x::InfiniteMPS, y::InfiniteMPS)
    Adata, Bdata = get_common_data(x, y)
    cell = TransferMatrix(Adata, Bdata)
    vl = random_left_boundary(cell)
    left_eigenvalue, left_eigenvector = _eigsolve_bare(x -> x * cell, vl)
    T = promote_type(scalartype(x), scalartype(y))
    if (T <: Real) && isa(left_eigenvalue, Complex)
        (abs(imag(left_eigenvalue)) < EIGENVALUE_IMAG_TOL) || @warn "imaginary part of eigenvalue is $(imag(left_eigenvalue))"
        left_eigenvalue = real(left_eigenvalue)
    end
    return left_eigenvalue
end
TK.norm(x::InfiniteMPS) = sqrt(abs(dot(x, x)))
# function TK.norm(x::InfiniteMPS; iscanonical::Bool=false)
#     if iscanonical
#         return norm(x[1]) / sqrt(dim(space_l(x)))
#     else
#         return sqrt(abs(dot(x, x)))
#     end
# end
