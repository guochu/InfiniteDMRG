function DMRG.expectation(psiA::InfiniteMPS, m::PartialMPO, psiB::InfiniteMPS, envs=environments(psiA, psiB))
	isempty(m) && return 0.
	pos = positions(m)
	ops = m.data
	pos_end = pos[end]
	util = get_trivial_leg(psiA[1])
	envl = leftenv(envs, pos[1])
	envr = rightenv(envs, pos_end)
	@tensor hold[-3 -2; -1] := conj(psiA[pos_end][-1, 1, 2]) * envr[3, 2] * psiB[pos_end][-3, 5, 3] * ops[end][-2, 1, 4, 5] * util[4]  
	for j in pos_end-1:-1:pos[1]
		pj = findfirst(x->x==j, pos)
		if isnothing(pj)
			hold = updateright(hold, psiA[j], pj, psiB[j])
		else
			hold = updateright(hold, psiA[j], ops[pj], psiB[j])
		end
	end
	@tensor r = conj(util[1]) * hold[2, 1, 3] * envl[3,2]
	# η = leading_eigenvalue(envs)
	# nperiod = num_period(pos[1], pos_end, unitcell_size(envs))
	return r  
end

"""
	expectation_canonical(m::PartialMPO, psi::InfiniteMPS)

This function requires iscanonical(psi) == true
"""
DMRG.expectation_canonical(m::PartialMPO, psi::InfiniteMPS) = DMRG._expectation_canonical(m, psi)
DMRG.expectation(m::PartialMPO, psi::InfiniteMPS, envs=environments(psi, psi)) = expectation(psi, m, psi, envs)
DMRG.expectationvalue(m::PartialMPO, psi::InfiniteMPS, envs=environments(psi, psi)) = expectation(m, psi, envs) / value(envs, m)

function num_period(_start::Int, _stop::Int, unitcellsize::Int)
	_start = r_start(_start, unitcellsize)
	_stop = r_stop(_stop, unitcellsize)
	@assert (_stop - _start+1) % unitcellsize == 0
	return div(_stop - _start+1, unitcellsize)
end
num_period(m::PartialMPO, unitcellsize::Int) = num_period(positions(m)[1], positions(m)[end], unitcellsize)
DMRG.value(x::InfiniteOverlapCache, m::PartialMPO) = value(x, num_period(m, unitcell_size(x)))
get_trivial_leg(m::AbstractTensorMap) = TensorMap(ones,scalartype(m),oneunit(space(m,1)), one(space(m,1)))