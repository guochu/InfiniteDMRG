struct OverlapCache{_O, _I, _L, _R, _T} <: AbstractInfiniteCache
	o::_O
	i::_I
	left::_L
	right::_R
	leading_eigenvalue::_T
end

TK.scalartype(::Type{OverlapCache{O, I, L, R, T}}) where {O, I, L, R, T} = promote_type(scalartype(O), scalartype(I))
unitcell_size(x::OverlapCache) = unitcell_size(x.o)
left_boundary(x::OverlapCache) = x.left
right_boundary(x::OverlapCache) = x.right
leading_eigenvalue(x::OverlapCache) = x.leading_eigenvalue

function DMRG.environments(A::M, B::M) where {M <: AbstractInfiniteTN}
	@assert unitcell_size(A) >= unitcell_size(B)
	return OverlapCache(A, B)
end

reset!(x::OverlapCache, vo, site::Int) = (x.o[site] = vo)

function unitcell_transfer_matrix(x::OverlapCache, start_pos::Int=1)
	hold = transfer_matrix(x.o[start_pos], x.i[start_pos])
	for n in start_pos+1:start_pos+unitcell_size(x)-1
		hold = updatecyclicleft(hold, x.o[n], x.i[n])
	end
	return hold
end

function unitcell_env(x::OverlapCache)
	heff = unitcell_transfer_matrix(x)
	return heff^(num_unitcells(x) - 1)
end


function cenv(x::OverlapCache, site::Int)
	N = unitcell_size(x)
	site = mod1(site, N)
	henv = unitcell_env(x)
	for l in N:-1:site+1
		henv = updatecyclicright(henv, x.o[l], x.i[l])
	end
	for l in 1:site-1
		henv = updatecyclicleft(henv, x.o[l], x.i[l])
	end
	return henv
	# return permute(henv, (3,1), (4,2))
end
ceff(x::OverlapCache, site::Int) = permute(cenv(x, site), (3,1), (4,2))

function calculate(env::OverlapCache) 
	heff = unitcell_transfer_matrix(env)
	n = heff^(num_unitcells(env))
	@tensor r = n[1,2,1,2]
	return r
end 
TK.dot(x::M, y::M) where {M <: AbstractPeriodicTN} = calculate(environments(x, y))
TK.norm(x::AbstractPeriodicTN) = sqrt(real(dot(x, x)))