"""
	struct MixedCanonicalInfiniteMPS
"""
struct MixedCanonicalInfiniteMPS{A <: MPSTensor, B <: MPSBondTensor} 
	AL::PeriodicArray{A, 1}
	CR::PeriodicArray{B, 1}
	AR::PeriodicArray{A, 1}
end

Base.length(a::MixedCanonicalInfiniteMPS) = length(a.AL)

function MixedCanonicalInfiniteMPS(AL::PeriodicArray{A, 1}) where {A <: MPSTensor}
    CR = [isomorphism(scalartype(A), space_l(item), space_l(item)) for item in AL]
    return MixedCanonicalInfiniteMPS(AL, CR, copy(AL))
end

function InfiniteMPS(x::MixedCanonicalInfiniteMPS; trunc::TruncationScheme=DefaultTruncation)
	return get_imps!(x.AL, deepcopy(x.CR), copy(x.AR), trunc)
end

