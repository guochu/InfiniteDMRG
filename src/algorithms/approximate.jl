function iterative_compress(x::InfiniteMPS, alg::DMRG1)
	y = _svd_guess!(deepcopy(x), alg.trunc)
	return iterative_compress!(y, x, alg)
end

function iterative_compress!(y::InfiniteMPS, x::InfiniteMPS, alg::DMRG1)
	y_new = _iterative_compress!(y, x, alg)
	y.data[:] = y_new.data
	y.svectors[:] = y_new.svectors
	return y
end

function _iterative_compress!(y::InfiniteMPS, x::InfiniteMPS, alg::DMRG1)
	@assert unitcell_size(y) == unitcell_size(x)
	tol = alg.tol

	AL = copy(y.data)
	AR = copy(y.data)
	# if svectors_uninitialized(y)
	# 	CR = PeriodicArray([TensorMap(ds->randn(scalartype(y), ds), space_l(item), space_l(item)) for item in AL])
	# else
	# 	CR = PeriodicArray([similar(y.s[i], scalartype(y)) for i in 1:length(AL)])
	# end
	CR = PeriodicArray([TensorMap(ds->randn(scalartype(y), ds), space_l(item), space_l(item)) for item in AL])
	

	# build initial random environment
	A2 = x.data
	left, right = random_environments(AL, CR, AR, A2)
	L = length(AL)

	delta = 2 * tol
	itr = 0
	while (itr < alg.maxiter) && (delta >= tol)
		CR_old = normalize!(CR[1])
		for j in 1:L
			AC_j = ac_prime(A2[j], left[j], right[j])
			AL[j], CR[j+1] = leftorth!(normalize!(AC_j), alg=QRpos())
			# normalize!(CR[j+1])
			Adata_i, Bdata_i = get_common_data(AL, A2, j, j)
			left[j+1] = left[j] * TransferMatrix(Adata_i, Bdata_i)
		end

		for j in L:-1:1
			AC_j = ac_prime(A2[j], left[j], right[j])
			CR[j], AR_j = rightorth(normalize!(AC_j), (1,), (2,3), alg=LQpos())
			# normalize!(CR[j])
			AR[j] = permute(AR_j, (1,2), (3,))
			Adata_i, Bdata_i = get_common_data(AR, A2, j, j)
			right[j-1] =TransferMatrix(Adata_i, Bdata_i) * right[j]
		end

		delta = norm(CR_old - CR[1])
		(alg.verbosity > 1) && println("Idmrg itr $(itr), err $(delta)")
		itr += 1
	end
	if (delta < tol) && (itr < alg.maxiter)
		(alg.verbosity > 1) && println("Idmrg converges in itr $(itr) sweeps")
	end
    if (alg.verbosity > 0) && (delta >= alg.tol)
        println("Idmrg fail to converge, required precision: $(alg.tol), actual precision $delta in $itr sweeps")
    end
 	# ismxiedcanonical(AL, CR, AR, tol=alg.toleig * 10000) || error("AL, CR, AR is not mixed-canonical")
	y_new = get_imps!(AL.data, CR.data, AR.data, alg.trunc)
	return y_new
end

function _svd_guess!(y::InfiniteMPS, trunc::TruncationScheme)
	for i in length(y):-1:1
		u, s, v = stable_tsvd(y[i], (1,), (2,3), trunc=trunc)
		y[i] = permute(v, (1,2), (3,))
		s = normalize!(s)
		u = u * s
		y.s[i] = s
		y[i-1] = @tensor tmp[1,2;4] := y[i-1][1,2,3] * u[3,4]
	end
	return y
end


# for mixee-canonical form, randomly initializing the environments
function random_environments(AL, CR, AR, A2) 
	# left boundary
	Adata, Bdata = get_common_data(AL, A2)
	cell = TransferMatrix(Adata, Bdata)
	vl = random_left_boundary(cell)

	L = length(Adata)
	left = Vector{typeof(vl)}(undef, L)
	left[1] = vl
	for i in 2:L
		Adata_i, Bdata_i = get_common_data(Adata, Bdata, i-1, i-1)
		left[i] = left[i-1] * TransferMatrix(Adata_i, Bdata_i)
	end

	# right boundary
	Adata, Bdata = get_common_data(AR, A2)
	cell = TransferMatrix(Adata, Bdata)
	vr = random_right_boundary(cell)
	right = Vector{typeof(vr)}(undef, L)
	right[end] = vr
	for i in L-1:-1:1
		Adata_i, Bdata_i = get_common_data(Adata, Bdata, i+1, i+1)
		right[i] = TransferMatrix(Adata_i, Bdata_i) * right[i+1]
	end

	return PeriodicArray(left), PeriodicArray(right)
end


