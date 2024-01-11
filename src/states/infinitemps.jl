abstract type AbstractInfiniteMPS{A <: MPSTensor} <: AbstractInfiniteTN{A} end
"""
	struct InfiniteMPS{A <: MPSTensor, B <: MPSBondTensor} 

The singular vector on the first site is important!!!
"""
struct InfiniteMPS{A <: MPSTensor, B <: MPSBondTensor} <: AbstractInfiniteMPS{A}
	data::PeriodicArray{A, 1}
	svectors::PeriodicArray{B, 1}

function InfiniteMPS(data::PeriodicArray{A, 1}, svectors::PeriodicArray{B, 1}) where {A<:MPSTensor, B<:DiagonalMap}
	@assert length(data) == length(svectors)
	check_mps_spaces(data)	
	return new{A, B}(data, svectors)
end

function InfiniteMPS(data::PeriodicArray{A, 1}, sl::MPSBondTensor) where {A<:MPSTensor}
	check_mps_spaces(data)	
	T = real(scalartype(A))
	B = bondtensortype(spacetype(A), Diagonal{T, Vector{T}})
	svectors = PeriodicArray{B, 1}(undef, length(data))
	svectors[1] = convert(B, sl) 
	for i in 2:length(data)
		svectors[i] = convert(B, id(space_l(data[i])))
	end
	return new{A, B}(data, svectors)
end 

end

InfiniteMPS(data::AbstractVector{<:MPSTensor}, svectors::AbstractVector{<:MPSBondTensor}) = InfiniteMPS(PeriodicArray(data), PeriodicArray(svectors))
InfiniteMPS(data::AbstractVector{<:MPSTensor}, sl::MPSBondTensor) = InfiniteMPS(PeriodicArray(data), sl)

function InfiniteMPS(data::AbstractVector{A}) where {A <: MPSTensor}
	T = real(scalartype(A))
	B = bondtensortype(spacetype(A), Diagonal{T, Vector{T}})
	sl = convert(B, id(space_l(data[1])))
	return 	InfiniteMPS(data, sl)
end

function Base.getproperty(psi::InfiniteMPS, s::Symbol)
	if s == :s
		return MPSBondView(psi)
	else
		return getfield(psi, s)
	end
end

function Base.setindex!(psi::InfiniteMPS, v::MPSTensor, i::Int)
	return setindex!(psi.data, v, i)
end 
Base.copy(psi::InfiniteMPS) = InfiniteMPS(copy(psi.data), copy(psi.svectors))

function Base.convert(::Type{<:InfiniteMPS}, psi::MPS)
	(space_l(psi) == space_r(psi)') || throw(SpaceMismatch("boundary space mismatch"))
	if svectors_uninitialized(psi)
		return InfiniteMPS(psi.data)
	else
		return InfiniteMPS(psi.data, psi.svectors[1:length(psi)])
	end
end

DMRG.isrightcanonical(a::InfiniteMPS; kwargs...) = all(x->isrightcanonical(x; kwargs...), a.data)
DMRG.isleftcanonical(a::InfiniteMPS; kwargs...) = all(x->isleftcanonical(x; kwargs...), a.data)
function iscanonical(a::InfiniteMPS; kwargs...)
	isrightcanonical(a) || return false
	S = a.s[1]
	hold = S' * l_LL(a, a) * S
	for i in 1:length(a)-1
		hold = updateleft(hold, a[i], a[i])
		tmp = a.s[i+1] * a.s[i+1]
		isapprox(hold, tmp; kwargs...) || return false
	end
	return true	
end


function check_mps_spaces(data::PeriodicArray)
	@assert !isempty(data)
	for i in 1:length(data)
		(space_r(data[i])' == space_l(data[i+1])) || throw(SpaceMismatch())
	end
end