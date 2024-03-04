DMRG.canonicalize(x::InfiniteMPS; alg::MatrixProductOrthogonalAlgorithm=InfiniteOrthogonalize(normalize=false)) = canonicalize(x, alg)
DMRG.canonicalize!(x::InfiniteMPS; alg::MatrixProductOrthogonalAlgorithm=InfiniteOrthogonalize(normalize=false)) = canonicalize!(x, alg)
DMRG.canonicalize!(x::InfiniteMPS, alg::Orthogonalize{TK.SVD}) = canonicalize!(x, InfiniteOrthogonalize(alg))
DMRG.canonicalize!(x::InfiniteMPS, alg::InfiniteOrthogonalize) = mixedcanonicalize2!(x, alg)
DMRG.canonicalize(x::InfiniteMPS, alg::InfiniteOrthogonalize) = mixedcanonicalize2(x, alg)

mixedcanonicalize2(x::InfiniteMPS, alg::InfiniteOrthogonalize) = mixedcanonicalize2!(copy(x), alg)



"""
	canonicalize!(x::InfiniteMPS; alg::Orthogonalize)
Preparing an infinite mps into (right-)canonical form
Reference: PHYSICAL REVIEW B 78, 155117 
"""
function mixedcanonicalize2!(x::InfiniteMPS, alg::InfiniteOrthogonalize)
	# alg.normalize || throw(ArgumentError("normalization has been doen for infinite mps"))
	# println("eta is ", eta)
	tolchol = alg.toleig * 10
	eta, Vl, Vr = leading_boundaries(x, tol=alg.toleig, maxiter=alg.maxitereig)

	Y = chol_split(Vl, tolchol)
	X = chol_split(Vr, tolchol)'
	U, S, V = stable_tsvd!(Y * X, trunc=alg.trunc)
	normalize!(S)
	
	# m = (S * V) / X
	m = U' * Y
	L = unitcell_size(x)
	for i in 1:(L-1)
		@tensor xj[1,3; 4] := m[1, 2] * x[i][2,3,4]
		x[i], m = leftorth!(xj, alg=TK.QRpos())
	end
	# m2 = Y \ (U * S)
	m2 = X * V'
	x[L] = @tensor tmp[1,3; 5] := m[1,2] * x[L][2,3,4] * m2[4,5]
	
	for i in L:-1:2
		v, ss, xj2, err = stable_tsvd(x[i], (1,), (2,3), trunc=alg.trunc)
		x[i] = permute(xj2, (1,2), (3,))
		normalize!(ss)
		x.s[i] = ss
		m = v * ss
		x[i-1] = @tensor tmp[1,2;4] := x[i-1][1,2,3] * m[3,4]
	end
	
	# Is there an easy way to improve the numeric stability of the following line of code?	
	x[1] = permute(S \ permute(x[1], (1,), (2,3)), (1,2), (3,))
	x.s[1] = S

	if !alg.normalize
		ηs = sqrt(real(eta))
		lmul!(ηs, x[1])
		lmul!(ηs, x.s[1])
	end

	return x
end




# """
# 	canonicalize!(x::InfiniteMPS; alg::Orthogonalize)
# Preparing an infinite mps into (right-)canonical form
# Reference: PHYSICAL REVIEW B 78, 155117 
# """
# function mixedcanonicalize2!(x::InfiniteMPS, alg::InfiniteOrthogonalize)
# 	# alg.normalize || throw(ArgumentError("normalization has been doen for infinite mps"))
# 	# println("eta is ", eta)
# 	tolchol = alg.toleig * 10
# 	eta, Vl, Vr = leading_boundaries(x, tol=alg.toleig, maxiter=alg.maxitereig)

# 	Y = chol_split(Vl, tolchol)
# 	X = chol_split(Vr, tolchol)'
# 	U, S, V = stable_tsvd!(Y * X, trunc=alg.trunc)
# 	println("here: ", norm(X), " ", norm(Y), " ", norm(S))
# 	alg.normalize && normalize!(S)
	
# 	# m = (S * V) / X
# 	m = U' * Y
# 	L = unitcell_size(x)
# 	for i in 1:(L-1)
# 		@tensor xj[1,3; 4] := m[1, 2] * x[i][2,3,4]
# 		x[i], m = leftorth!(xj, alg=TK.QRpos())
# 	end
# 	# m2 = Y \ (U * S)
# 	m2 = X * V'
# 	x[L] = @tensor tmp[1,3; 5] := m[1,2] * x[L][2,3,4] * m2[4,5]
	
# 	for i in L:-1:2
# 		# @tensor xj[1; 2 4] := x[i][1,2,3] * m[3,4]
# 		v, ss, xj2, err = stable_tsvd(x[i], (1,), (2,3), trunc=alg.trunc)
# 		x[i] = permute(xj2, (1,2), (3,))
# 		alg.normalize && (normalize!(ss))
# 		x.s[i] = ss
# 		m = v * ss
# 		x[i-1] = @tensor tmp[1,2;4] := x[i-1][1,2,3] * m[3,4]
# 	end
	
# 	# Is there an easy way to improve the numeric stability of the following line of code?	
# 	x[1] = permute(S \ permute(x[1], (1,), (2,3)), (1,2), (3,))
# 	x.s[1] = S

# 	# # Is it better to truncate the first site again?
# 	# v, ss, xj2, err = stable_tsvd(x[1], (1,), (2,3), trunc=alg.trunc)
# 	# if alg.normalize
# 	# 	normalize!(ss)
# 	# 	x[1] = permute(xj2, (1,2), (3,))
# 	# 	x.s[1] = ss
# 	# else
# 	# 	n = norm(ss) / norm(S)
# 	# 	x[1] = lmul!(n, permute(xj2, (1,2), (3,)))
# 	# 	x.s[1] = lmul!(1/n, ss) 
# 	# end
# 	# x[end] = x[end] * v
# 	return x
# end