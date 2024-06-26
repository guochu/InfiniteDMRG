println("------------------------------------")
println("|           Infinite MPS           |")
println("------------------------------------")

@testset "Infinite MPS initializer: product state and random state" begin
	VZ2 = Rep[ℤ₂](0=>1, 1=>1)
	physpaces = [VZ2 for i in 1:4]
	psi = prodimps(ComplexF64, [VZ2 for i in 1:4], [0, 1, 1, 0])
	@test length(psi) == 4
	for i in 1:length(psi)
		@test !isdual(space(psi[i], 1))
		@test !isdual(space(psi[i], 2))
		@test isdual(space(psi[i], 3))
	end
	canonicalize!(psi)
	for i in (1, length(psi)+1)
		@test !isdual(space(psi.s[i], 1))
		@test isdual(space(psi.s[i], 2))
	end
	@test psi.s[1] == one(psi.s[1])
	@test psi.s[end] == one(psi.s[end])
	@test scalartype(psi) == ComplexF64
	@test bond_dimensions(psi) == [1,1,1,1]
	@test space_l(psi) == oneunit(VZ2)
	@test space_r(psi) == oneunit(VZ2)'
	@test physical_spaces(psi) == physpaces

	psi2 = prodimps(ComplexF64, [VZ2 for i in 1:4], [0, 1, 0, 1])
	@constinferred psi + psi2



	# random mps, trivial sector
	VSU2 = Rep[U₁×SU₂]((-0.5, 0)=>1, (0.5, 0)=>1, (0, 0.5)=>1)
	vspace = Rep[U₁×SU₂]((0, 0)=>4, (0, 1)=>4, (0, 0.5)=>4)
	orth = Orthogonalize(trunc=truncdimcutoff(D=200, ϵ=1.0e-8, add_back=0), normalize=true)
	for T in (Float64, ComplexF64)
		psi = InfiniteMPS(randn, T, [VSU2 for i in 1:4], [vspace for i in 1:4])
		canonicalize!(psi, alg=orth)
		@test isrightcanonical(psi)
		@test iscanonical(psi)
	end

end


@testset "Infinite MPS: Orthogonalization" begin
	tol = 1.0e-7
	for L in 2:4
		pspace = Rep[ℤ₂](0=>1, 1=>1)
		vspace = Rep[ℤ₂](0=>4, 1=>4)

		mps = InfiniteMPS(randn, Float64, [pspace for i in 1:L], vspace)
		orth = InfiniteOrthogonalize(Orthogonalize(trunc = truncdimcutoff(D=200, ϵ=1.0e-8, add_back=0), normalize=true))
		mps0 = copy(mps)
		canonicalize!(mps0, orth)

		@test abs(norm(mps0)-1) < tol
		@test !isleftcanonical(mps0)
		@test isrightcanonical(mps0)
		@test iscanonical(mps0)

		mps = InfiniteMPS(mixedcanonicalize(mps, orth), trunc=orth.trunc)

		@test abs(norm(mps)-1) < tol
		@test !isleftcanonical(mps)
		@test isrightcanonical(mps)
		@test iscanonical(mps)

		@test distance(mps, mps0) < tol
	end

end