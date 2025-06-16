
function compress(x::MixedCanonicalInfiniteMPS, trunc::TruncationScheme)
	USV = map(1:length(x)) do i
		stable_tsvd(x.CR[i], trunc=trunc)
	end
	AR = map(1:length(x)) do i
		@tensor tmp[1,3;5] := USV[i][3][1,2] * x.AR[i][2,3,4] * USV[mod1(i+1,end)][3]'[4,5]
	end
	AL = map(1:length(x)) do i
		@tensor tmp[1,3;5] := USV[i][1]'[1,2] * x.AL[i][2,3,4] * USV[mod1(i+1,end)][1][4,5]
	end
	CR = map(i->USV[i][2], 1:length(USV))
	return MixedCanonicalInfiniteMPS(AL, CR, AR)
end

function fixed_environments(x::InfiniteMPS, y::MixedCanonicalInfiniteMPS; alg=Arnoldi())
	# Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)
	AL, CR, AR = y.AL, y.CR, y.AR
	A2 = x.data

	# left boundary
	Adata, Bdata = get_common_data(AL, A2)
	cell = TransferMatrix(Adata, Bdata)
	vl = random_left_boundary(cell)
	(vals, v0) = fixedpoint(x -> x*cell, vl, :LM, alg)

	L = length(Adata)
	left = Vector{typeof(v0)}(undef, L)
	left[1] = v0
	for i in 2:L
		Adata_i, Bdata_i = get_common_data(Adata, Bdata, i-1, i-1)
		left[i] = left[i-1] * TransferMatrix(Adata_i, Bdata_i)
	end

	# right boundary
	Adata, Bdata = get_common_data(AR, A2)
	cell = TransferMatrix(Adata, Bdata)
	vr = random_right_boundary(cell)
	(vals, v0) = fixedpoint(x -> cell * x, vr, :LM, alg)

	right = Vector{typeof(v0)}(undef, L)
	right[end] = v0
	for i in L-1:-1:1
		Adata_i, Bdata_i = get_common_data(Adata, Bdata, i+1, i+1)
		right[i] = TransferMatrix(Adata_i, Bdata_i) * right[i+1]
	end

	return PeriodicArray(left), PeriodicArray(right)
end

function iterative_compress!(x::InfiniteMPS, y::MixedCanonicalInfiniteMPS, alg::DMRG1)
	L = length(x)
	@assert length(x) == length(y)
	tol = alg.tol

	T = scalartype(x)
	AL, AR, CR = y.AL, y.AR, PeriodicArray([copy!(similar(i, T), i) for i in y.CR])
	A2 = x.data
	left, right = fixed_environments(x, y)

	delta = 2 * tol
	itr = 0
	errs = Float64[]
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
		push!(errs, delta)
		(alg.verbosity > 2) && println("Idmrg itr $(itr), err $(delta)")
		itr += 1
	end
	if (alg.verbosity >= 3)
        pic = plot(log.(errs))
        show(pic)
        println()
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



function iterative_compress(x::InfiniteMPS, alg1::InfiniteOrthogonalize, alg2::DMRG1)
	x2 = mixedcanonicalize(x, alg1)
	x3 = compress(x2, alg1.trunc)
	return iterative_compress!(x, x3, alg2) 
end


