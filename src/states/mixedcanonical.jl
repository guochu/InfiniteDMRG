"""
	struct MixedCanonicalInfiniteMPS
"""
struct MixedCanonicalInfiniteMPS{A <: MPSTensor, B <: MPSBondTensor} 
	AL::PeriodicArray{A, 1}
	CR::PeriodicArray{B, 1}
	AR::PeriodicArray{A, 1}
end

Base.length(a::MixedCanonicalInfiniteMPS) = length(a.AL)

MixedCanonicalInfiniteMPS(AL::AbstractVector{A}, CR::AbstractVector{B}, AR::AbstractVector{A}) where {A <: MPSTensor, B <: MPSBondTensor} = MixedCanonicalInfiniteMPS(
							PeriodicArray(AL), PeriodicArray(CR), PeriodicArray(AR))

Base.copy(x::MixedCanonicalInfiniteMPS) = MixedCanonicalInfiniteMPS(copy(x.AL), copy(x.CR), copy(x.AR))

# function MixedCanonicalInfiniteMPS(AL::PeriodicArray{A, 1}) where {A <: MPSTensor}
#     CR = [isomorphism(scalartype(A), space_l(item), space_l(item)) for item in AL]
#     return MixedCanonicalInfiniteMPS(AL, CR, copy(AL))
# end

function InfiniteMPS(x::MixedCanonicalInfiniteMPS; trunc::TruncationScheme=DefaultTruncation)
	return get_imps!(x.AL.data, deepcopy(x.CR.data), copy(x.AR.data), trunc)
end


function MixedCanonicalInfiniteMPS(x::InfiniteMPS)
	AL = x.data
	AR = copy(AL)
	CR = PeriodicArray([copy!(similar(s, scalartype(x)), s) for s in x.svectors])
	return MixedCanonicalInfiniteMPS(AL, CR, AR)
end


function get_imps!(AL, CR, AR, trunc::TruncationScheme)
    (length(AL) == length(CR) == length(AR)) || throw(DimensionMismatch())
    U1, S1, V1o = stable_tsvd!(CR[1], trunc=trunc)
    V1 = V1o
    svectors = Vector{typeof(S1)}(undef, length(CR))
    svectors[1] = S1
    for loc in 2:length(CR)
        U2, S2, V2 = stable_tsvd!(CR[loc], trunc=trunc)
        @tensor tmp[1,3;5] := V1[1,2] * AR[loc-1][2,3,4] * V2'[4,5]
        svectors[loc] = S2
        AR[loc-1] = tmp
        V1 = V2
    end
    @tensor tmp[1,3;5] := V1[1,2] * AR[end][2,3,4] * V1o'[4,5]
    AR[end] = tmp
    return InfiniteMPS(AR, svectors)
end