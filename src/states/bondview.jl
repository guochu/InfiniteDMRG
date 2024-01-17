
struct MPSBondView{M}
	parent::M
end

Base.getindex(psi::MPSBondView, i::Int) = getindex(psi.parent.svectors, i)
Base.firstindex(m::MPSBondView) = firstindex(m.parent.svectors)
Base.lastindex(m::MPSBondView) = lastindex(m.parent.svectors)
function Base.setindex!(m::MPSBondView, v, i::Int)
	return setindex!(m.parent.svectors, v, i)
end
