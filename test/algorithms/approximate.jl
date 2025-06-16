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

	alg = DMRG1(D=100, verbosity=0)
	state = iterative_compress(state, alg)	

	observers = [PartialMPO([sp, sp'], [1, 1+i]) for i in 1:10]
	obs = [expectation_canonical(ob, state) for ob in observers]
	for i in 1:10
		state = mpo * state
		state = iterative_compress(state, alg)	
		append!(obs, [expectation_canonical(ob, state) for ob in observers])
	end	
	return state, obs
end


function infinite_xxz_mpo_iterative2()
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

	trunc = truncdimcutoff(D=100, ϵ=1.0e-8)
	state = iterative_compress(state, InfiniteOrthogonalize(trunc=trunc), DMRG1())

	observers = [PartialMPO([sp, sp'], [1, 1+i]) for i in 1:10]
	obs = [expectation_canonical(ob, state) for ob in observers]
	for i in 1:10
		state = mpo * state
		state = iterative_compress(state, InfiniteOrthogonalize(trunc=trunc), DMRG1())	
		append!(obs, [expectation_canonical(ob, state) for ob in observers])
	end	
	return state, obs
end


@testset "Approximate: SVD compression vs iterative compression" begin
    state1, obs1 = infinite_xxz_mpo_svd()
    # println(obs1)
    state2, obs2 = infinite_xxz_mpo_iterative()
    # println(obs2)
    state3, obs3 = infinite_xxz_mpo_iterative2()
    # println(obs3)
    @test abs(norm(state1)-1) < 1.0e-6
    @test abs(norm(state2)-1) < 1.0e-6
    @test abs(norm(state3)-1) < 1.0e-6
    @test distance(state1, state2) < 1.0e-4
    @test distance(state1, state3) < 1.0e-4
    @test norm(obs1 - obs2) / norm(obs1) < 1.0e-2
    @test norm(obs1 - obs3) / norm(obs1) < 1.0e-2
end
