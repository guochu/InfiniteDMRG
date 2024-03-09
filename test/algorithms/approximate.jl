println("------------------------------------")
println("--------|    Approximate    |-------")
println("------------------------------------")


function infinite_xxz_mpo_svd()
	p = spin_site_ops_u1x()	
	sp, sm, sz = p["+"], p["-"], p["z"]
	pspace = physical_space(sp)

	hz = 0.7
	Jzz = 1.3

	m = fromABCD(C=[2*sp, 2*sm, Jzz*sz], B= [sp', sm', sz], D=hz*sz)
	h = MPOHamiltonian([m])
	U = timeevompo(h, -0.01im, WII())

	mpo = InfiniteMPO(U)

	initial_state = [-0.5, 0.5]
	state = prodimps(ComplexF64, [pspace for i in 1:length(initial_state)], initial_state)
	orth = Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8, add_back=0), normalize=true)
	canonicalize!(state, alg=orth)	

	observers = [PartialMPO([sp, sp'], [1, 1+i]) for i in 1:10]
	obs = [expectation_canonical(ob, state) for ob in observers]
	for i in 1:10
		state = mpo * state
		canonicalize!(state, alg=orth)
		append!(obs, [expectation_canonical(ob, state) for ob in observers])
	end	
	return state, obs
end


function infinite_xxz_mpo_iterative()
	p = spin_site_ops_u1x()	
	sp, sm, sz = p["+"], p["-"], p["z"]
	pspace = physical_space(sp)

	hz = 0.7
	Jzz = 1.3

	m = fromABCD(C=[2*sp, 2*sm, Jzz*sz], B= [sp', sm', sz], D=hz*sz)
	h = MPOHamiltonian([m])
	U = timeevompo(h, -0.01im, WII())

	mpo = InfiniteMPO(U)

	initial_state = [-0.5, 0.5]
	state = prodimps(ComplexF64, [pspace for i in 1:length(initial_state)], initial_state)

	alg = DMRG1(D=100, verbosity=2)
	state = iterative_compress(state, alg)	

	observers = [PartialMPO([sp, sp'], [1, 1+i]) for i in 1:10]
	obs = [expectation_canonical(ob, state) for ob in observers]
	for i in 1:10
		state = mpo * state
		# D = max(alg.D, bond_dimension(state))
		# state = iterative_compress!(randomimps(scalartype(state), physical_spaces(state), D=D), state, alg)	
		state = iterative_compress(state, alg)	
		append!(obs, [expectation_canonical(ob, state) for ob in observers])
	end	
	return state, obs
end

@testset "TimeEvoMPO: compared with finite system" begin
    state1, obs1 = infinite_xxz_mpo_svd()
    # println(obs1)
    state2, obs2 = infinite_xxz_mpo_iterative()
    # println(obs2)
    @test distance(state1, state2) < 1.0e-4
    @test norm(obs1 - obs2) / norm(obs1) < 1.0e-2
end
