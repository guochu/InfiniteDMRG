println("------------------------------------")
println("--------|    Expectation    |-------")
println("------------------------------------")



@testset "Expectation value" begin
	pspaces = [Rep[ℤ₂](0=>1, 1=>1), Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)]
	vspaces = [Rep[ℤ₂](0=>4, 1=>4), Rep[U₁×SU₂]((0, 0)=>4, (0, 1)=>4, (0, 0.5)=>4)]

	trunc = truncdimcutoff(D=50, ϵ=1.0e-8, add_back=0)

	for (pspace, vspace) in zip(pspaces, vspaces)
		for T in (Float64, ComplexF64)
			psi = InfiniteMPS(randn, T, [pspace for i in 1:4], [vspace for i in 1:4])
			sp = randn(T, oneunit(vspace) ⊗ pspace, vspace ⊗ pspace)
			observers = [PartialMPO([sp, sp'], [1, i+1]) for i in 1:30]
			# observables
			envs = environments(psi, psi)
			obs1 = [expectationvalue(ob, psi, envs) for ob in observers]
			# println(obs1)

			psi2 = canonicalize!(copy(psi), alg = Orthogonalize(trunc=trunc, normalize=false))
			envs = environments(psi2, psi2)
			obs2 = [expectationvalue(ob, psi2, envs) for ob in observers]
			@test abs(dot(psi, psi) - dot(psi2, psi2)) < 1.0e-8
			@test maximum(abs, obs1 - obs2) < 1.0e-8

			psi3 = canonicalize!(copy(psi), alg = Orthogonalize(trunc=trunc, normalize=true))
			obs3 = [expectation_canonical(ob, psi3) for ob in observers]
			@test abs(dot(psi3, psi3)-1) < 1.0e-8
			@test maximum(abs, obs1 - obs3) < 1.0e-8
		end
	end

	
end