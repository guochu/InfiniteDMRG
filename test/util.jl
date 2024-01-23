function spin_site_ops_u1x()
    ph = Rep[U₁](-0.5=>1, 0.5=>1)
    vacuum = oneunit(ph)
    σ₊ = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](1=>1) ⊗ ph)
    blocks(σ₊)[Irrep[U₁](0.5)] = ones(1, 1)
    σ₋ = TensorMap(zeros, vacuum ⊗ ph ← Rep[U₁](-1=>1) ⊗ ph)
    blocks(σ₋)[Irrep[U₁](-0.5)] = ones(1, 1)
    σz = TensorMap(ones, ph ← ph)
    blocks(σz)[Irrep[U₁](-0.5)] = -ones(1, 1)
    return Dict("+"=>σ₊, "-"=>σ₋, "z"=>σz)
end