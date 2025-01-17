println("------------------------------------")
println("|           Infinite MPO           |")
println("------------------------------------")

function spin_site_ops_u1x()
    ph = Rep[U₁](-0.5=>1, 0.5=>1)
    vacuum = oneunit(ph)
    σ₊ = zeros(vacuum ⊗ ph ← Rep[U₁](1=>1) ⊗ ph)
    copy!(block(σ₊, Irrep[U₁](0.5)), ones(1, 1))
    σ₋ = zeros(vacuum ⊗ ph ← Rep[U₁](-1=>1) ⊗ ph)
    copy!(block(σ₋, Irrep[U₁](-0.5)), ones(1, 1))
    σz = ones(ph ← ph)
    copy!(block(σz, Irrep[U₁](-0.5)), -ones(1, 1))
    return Dict("+"=>σ₊, "-"=>σ₋, "z"=>σz)
end

@testset "Infinite MPO initializer: product operator" begin
	# u1 symmetry
	p = spin_site_ops_u1x()
	sp, sm, z = p["+"], p["-"], p["z"]
	ph = space(z, 1)
	physpaces = [ph for i in 1:4]

	# test adjoint expectation
	L = 4
	physpaces = [ph for i in 1:L]
	mpo = randomimpo(ComplexF64, physpaces, D=10)
	mps = randomimps(ComplexF64, physpaces, D=4)

	@constinferred mpo * mps

end