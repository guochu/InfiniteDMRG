

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


function get_common_data(a::AbstractInfiniteTN, b::AbstractInfiniteTN)
    cellsize = max(unitcell_size(a), unitcell_size(b))
    Adata = [a[i] for i in 1:cellsize]
    Bdata = [b[i] for i in 1:cellsize]
    return Adata, Bdata
end