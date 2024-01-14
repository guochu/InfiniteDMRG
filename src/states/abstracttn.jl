abstract type AbstractInfiniteMPS{M <: MPSTensor} <: AbstractMPS{M} end

Base.length(m::AbstractInfiniteMPS) = length(storage(m))
unitcell_size(m::AbstractInfiniteMPS) = length(m)
