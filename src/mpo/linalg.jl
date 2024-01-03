

"""
    Base.:+(hA::InfiniteMPO, hB::InfiniteMPO) 
    addition of two MPOs
"""
function Base.:+(hA::InfiniteMPO, hB::InfiniteMPO)
	Adata, Bdata = get_common_data(hA, hB)
	cellsize = length(Adata)
    T = promote_type(scalartype(hA), scalartype(hB))
    S = spacetype(hA)
    M = mpotensortype(S, T)
    embedders = PeriodicArray([left_embedders(T, space_l(aj), space_l(bj)) for (aj, bj) in zip(Adata, Bdata)])
    r = Vector{M}(undef, cellsize)
    for i in 1:cellsize
        @tensor m1[-1 -2; -3 -4] := embedders[i-1][1][-1, 1] * hA[i][1,-2,2,-4] * (embedders[i][1])'[2, -3]
        @tensor m1[-1 -2; -3 -4] += embedders[i-1][2][-1, 1] * hB[i][1,-2,2,-4] * (embedders[i][2])'[2, -3]
        r[i] = m1 
    end
    return InfiniteMPO(r)
end

function Base.:*(h::InfiniteMPO, psi::InfiniteMPS)
	Adata, Bdata = get_common_data(h, psi)
	cellsize = length(Adata)
    r = [@tensor tmp[-1 -2; -3 -4 -5] := a[-1, -3, -4, 1] * b[-2, 1, -5] for (a, b) in zip(Adata, Bdata)]
    fusion_ts = PeriodicArray([isomorphism(space(item, 4)' ⊗ space(item, 5)', fuse(space(item, 4)', space(item, 5)')) for item in r])
    M = mpstensortype(spacetype(h), promote_type(scalartype(h), scalartype(psi)))
    mpstensors = Vector{M}(undef, cellsize)
    for i in 1:cellsize
        @tensor tmp[-1 -2; -3] := conj(fusion_ts[i-1][1,2,-1]) * r[i][1,2,-2,3,4] * fusion_ts[i][3,4,-3]
        mpstensors[i] = tmp
    end
    return InfiniteMPS(mpstensors)
end

function Base.:*(a::InfiniteMPO, b::InfiniteMPO) 
	Adata, Bdata = get_common_data(a, b)
	cellsize = length(Adata)
    r = [@tensor tmp[-1 -2 -3; -4 -5 -6] := aj[-1, -3, -4, 1] * bj[-2, 1, -5, -6] for (aj, bj) in zip(Adata, Bdata)]
    fusion_ts = PeriodicArray([isomorphism(fuse(space(item, 1), space(item, 2)), space(item, 1) ⊗ space(item, 2)) for item in r])
    M = mpotensortype(spacetype(a), promote_type(scalartype(a), scalartype(b)))
    mpotensors = Vector{M}(undef, cellsize)
    for i in 1:cellsize
        @tensor tmp[-1 -2; -3 -4] := fusion_ts[i][-1,1,2] * r[i][1,2,-2,3,4,-4] * conj(fusion_ts[i+1][-3,3,4])
        mpotensors[i] = tmp
    end   
    
    return InfiniteMPO(mpotensors)
end