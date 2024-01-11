

function InfiniteMPS(f, ::Type{T}, physpaces::AbstractVector{S}, virtualpaces::AbstractVector{S}) where {T <: Number, S <: ElementarySpace}
	@assert length(physpaces) == length(virtualpaces)
	any(x -> dim(x)==0, virtualpaces) &&  @warn "auxiliary space is empty"
	virtualpaces2 = PeriodicArray(virtualpaces)
	data = [TensorMap(f, T, virtualpaces2[i] ⊗ physpaces[i] ← virtualpaces2[i+1]) for i in 1:length(physpaces)]
	return InfiniteMPS(data)
end

function InfiniteMPS(f, ::Type{T}, physpaces::AbstractVector{S}, maxvirtualspace::S; right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = right
	for i in 2:L
		virtualpaces[i] = infimum(fuse(virtualpaces[i-1], physpaces[i-1]), maxvirtualspace)
	end
	virtualpaces[L+1] = right
	for i in L-1:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(physpaces[i]', virtualpaces[i+1]))
	end
	return InfiniteMPS(f, T, physpaces, virtualpaces[1:L])
end


function prodimps(::Type{T}, physpaces::AbstractVector{S}, physectors::AbstractVector; right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	L = length(physpaces)
	(L == length(physectors)) || throw(DimensionMismatch())
	physectors = [convert(sectortype(S), item) for item in physectors]

	# the total quantum number is ignored in the Abelian case
	if FusionStyle(sectortype(S)) isa UniqueFusion
		rightind, = ⊗(physectors...)
		right = S((rightind=>1,))
		(right == oneunit(right)) || throw(ArgumentError("right space should be vacuum by convention"))
	end
	virtualpaces = Vector{S}(undef, L+1)
	virtualpaces[1] = right
	for i in 2:L
		virtualpaces[i] = fuse(virtualpaces[i-1], S((physectors[i-1]=>1,)) )
	end
	virtualpaces[L+1] = right
	for i in L:-1:2
		virtualpaces[i] = infimum(virtualpaces[i], fuse(virtualpaces[i+1],  S((physectors[i]=>1,))' ))
	end
	return InfiniteMPS(ones, T, physpaces, virtualpaces[1:L])
end
prodimps(::Type{T}, physpace::S, physectors::AbstractVector; kwargs...) where {T <: Number, S <: ElementarySpace} = prodimps(T, [physpace for i in 1:length(physectors)], physectors; kwargs...)
prodimps(physpaces::AbstractVector{S}, physectors::AbstractVector; kwargs...) where {S <: ElementarySpace} = prodimps(Float64, physpaces, physectors; kwargs...)
prodimps(physpace::S, physectors::AbstractVector; kwargs...) where {S <: ElementarySpace} = prodimps(Float64, physpace, physectors; kwargs...)


"""
	randomimps(::Type{T}, physpaces::Vector{S}; D::Int, left::S, right::S)

Return a random infinite MPS.

Each virtual space has multiplicity 1 to allow the largest number of different 
quantum numbers to be explored under bond dimension D
"""
function randomimps(::Type{T}, physpaces::AbstractVector{S}; D::Int, right::S=oneunit(S)) where {T <: Number, S <: ElementarySpace}
	virtualpaces = DMRG._max_mps_virtual_spaces(physpaces, D, right, right)
	r = InfiniteMPS(randn, T, physpaces, virtualpaces[1:length(physpaces)])
	canonicalize!(r)
	return r
end
randomimps(physpaces::AbstractVector{S}; kwargs...) where {S <: ElementarySpace} = randomimps(Float64, physpaces; kwargs...)
