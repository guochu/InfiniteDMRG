abstract type AbstractInfiniteTN{M} end
storage(m::AbstractInfiniteTN) = m.data
Base.length(m::AbstractInfiniteTN) = length(storage(m))
unitcell_size(m::AbstractInfiniteTN) = length(storage(m))
TK.scalartype(::Type{<:AbstractInfiniteTN{M}}) where M = scalartype(M)
TK.spacetype(::Type{<:AbstractInfiniteTN{M}}) where M = spacetype(M)
TK.spacetype(x::AbstractInfiniteTN) = spacetype(typeof(x))
Base.getindex(m::AbstractInfiniteTN, i::Int) = getindex(storage(m), i)
Base.setindex!(m::AbstractInfiniteTN, v, i::Int) = setindex!(storage(m), v, i)
DMRG.bond_dimension(m::AbstractInfiniteTN, i::Int) = dim(space_r(m[i]))
DMRG.bond_dimensions(m::AbstractInfiniteTN) = [bond_dimension(m, i) for i in 1:length(m)]
DMRG.space_l(m::AbstractInfiniteTN) = space_l(m[1])
DMRG.space_r(m::AbstractInfiniteTN) = space_r(m[length(m)])
DMRG.physical_space(m::AbstractInfiniteTN, i::Int) = physical_space(m[i])
DMRG.physical_spaces(m::AbstractInfiniteTN) = [physical_space(m, i) for i in 1:length(m)]
virtual_space(m::AbstractInfiniteTN, i::Int) = space_l(m[i])
virtual_spaces(m::AbstractInfiniteTN) = [virtual_space(m, i) for i in 1:length(m)]

