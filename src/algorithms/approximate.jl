
function approximate!(y::InfiniteMPS, x::InfiniteMPS)
	@assert unitcell_size(y) == unitcell_size(x)
	A = 
end



"
solves AL * C = C * A in-place
"
function uniform_leftorth!(AL, CR, A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
    #(_,CR[end]) = leftorth!(CR[end], alg = TensorKit.QRpos());
    normalize!(CR[end])

    iteration = 1
    delta = 2 * tol

    while iteration < maxiter && delta > tol
        if iteration > 10 #when qr starts to fail, start using eigs - should be kw arg
            alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)

            (vals, vecs) = eigsolve(TransferMatrix(A, AL), CR[end], 1, :LM, alg)
            (_, CR[end]) = leftorth!(vecs[1]; alg=TK.QRpos())
        end

        cold = CR[end]

        for loc in 1:length(AL)
            @tensor tmp[1,3;4] = CR[mod1(loc - 1, end)][1,2] * A[loc][2,3,4]
            AL[loc], CR[loc] = leftorth!(tmp; alg=TK.QRpos())
            normalize!(CR[loc])
        end

        #update delta
        if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
            delta = norm(cold - CR[end])
        end

        iteration += 1
    end

    delta > tol && @warn "leftorth failed to converge $(delta)"

    return AL, CR
end

"
solves C * AR = C * A in-place
"
function uniform_rightorth!(AR, CR, A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
    #(CR[end],_) = rightorth!(CR[end], alg = TensorKit.LQpos());
    normalize!(CR[end])

    iteration = 1
    delta = 2 * tol
    while iteration < maxiter && delta > tol
        if iteration > 10#when qr starts to fail, start using eigs
            alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            (vals, vecs) = eigsolve(TransferMatrix(A, AR), CR[end], 1, :LM, alg)
            (CR[end], _) = rightorth!(vecs[1]; alg=TK.LQpos())
        end

        cold = CR[end]
        for loc in length(AR):-1:1
            AR[loc] = A[loc] * CR[loc]

            CR[mod1(loc - 1, end)], tmp = rightorth!(permute(AR[loc], (1,), (2,3)); alg=TK.LQpos())
            AR[loc] = permute(tmp, (1,2), (3,))
            normalize!(CR[mod1(loc - 1, end)])
        end

        #update counters and delta
        if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
            delta = norm(cold - CR[end])
        end

        iteration += 1
    end

    delta > tol && @warn "rightorth failed to converge $(delta)"

    return AR, CR
end
