abstract type AbstractInfiniteMPO{A <: MPOTensor} <: AbstractMPO{A} end
unitcell_size(m::AbstractInfiniteMPO) = length(m)

struct InfiniteMPO{M<:MPOTensor} <: AbstractInfiniteMPO{M}
	data::PeriodicArray{M, 1}

function InfiniteMPO(data::PeriodicArray{M, 1}) where {M<:MPOTensor}
	check_mpo_spaces(data)
	return new{M}(data)
end

end


InfiniteMPO(data::AbstractVector{<:MPOTensor}) = InfiniteMPO(PeriodicArray(data))

storage(a::InfiniteMPO) = a.data
Base.copy(h::InfiniteMPO) = InfiniteMPO(storage(h))
Base.length(a::InfiniteMPO) = length(storage(a))
Base.isempty(a::InfiniteMPO) = isempty(storage(a))
Base.getindex(a::InfiniteMPO, i::Int) = getindex(storage(a), i)
Base.firstindex(a::InfiniteMPO) = firstindex(storage(a))
Base.lastindex(a::InfiniteMPO) = lastindex(storage(a))

function Base.complex(psi::InfiniteMPO)
	if scalartype(psi) <: Real
		data = [complex(item) for item in psi.data]
		return InfiniteMPO(data)
	end
	return psi
end

function check_mpo_spaces(mpotensors::PeriodicArray)
	@assert !isempty(mpotensors)
	# all(check_mpotensor_dir, mpotensors) || throw(SpaceMismatch())
	for i in 1:length(mpotensors)
		(space_r(mpotensors[i]) == space_l(mpotensors[i+1])') || throw(SpaceMismatch())
	end
end

Base.convert(::Type{<:InfiniteMPO}, h::MPO) = InfiniteMPO(h.data)