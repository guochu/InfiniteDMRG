println("------------------------------------")
println("--------|    TimeEvoMPO    |--------")
println("------------------------------------")


function finite_xxz_mpo()
	L = 100
	p = spin_site_ops_u1x()	
	sp, sm, sz = p["+"], p["-"], p["z"]
	pspace = physical_space(sp)
	physpaces = [pspace for i in 1:L]

	hz = 0.7
	Jzz = 1.3

	m = fromABCD(C=[2*sp, 2*sm, Jzz*sz], B= [sp', sm', sz], D=hz*sz)
	Uj = timeevompo(m, -0.01im, WII())
	U = vcat([Uj[1, :]], [Uj for i in 1:L-2], [Uj[:, 1]])
	mpo = MPO(MPOHamiltonian(U))
	# h = MPOHamiltonian([m for i in 1:L])
	# mpo = MPO(timeevompo(h, -0.01im, WII()))

	initial_state = [-0.5 for i in 1:L]
	for i in 2:2:L
    	initial_state[i] = 0.5
	end

	state = prodmps(ComplexF64, physpaces, initial_state)
	observers = [PartialMPO([sp, sp'], [45, 45+i]) for i in 1:10]

	orth = Orthogonalize(trunc=truncdimcutoff(D=100, ϵ=1.0e-8, add_back=0), normalize=false)
	canonicalize!(state, alg=orth)	

	envs = environments(state, state)
	obs = [expectation(ob, state, envs) / value(envs) for ob in observers]

	for i in 1:10
		state = mpo * state
		canonicalize!(state, alg=orth)
		envs = environments(state, state)
		append!(obs, [expectation(ob, state, envs) / value(envs) for ob in observers])
	end	
	return obs
end

function infinite_xxz_mpo()
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
	return obs
end

@testset "TimeEvoMPO: compared with finite system" begin
    obs1 = finite_xxz_mpo()
    # println(obs1)
    obs2 = infinite_xxz_mpo()
    # println(obs2)
    @test norm(obs1 - obs2) / norm(obs1) < 1.0e-2
end
