# function mixedcanonicalize!(x::InfiniteMPS, alg::InfiniteOrthogonalize)
#     r = mixedcanonicalize(x, alg)
#     x.data[:] = r.data
#     x.svectors[:] = r.svectors
#     return x
# end

function ismxiedcanonical(AL, CR, AR; tol::Real=Defaults.tolgauge)
    (length(AL) == length(CR) == length(AR)) || throw(ArgumentError("AL, CR, AR size mismatch"))
    for loc in 1:length(AL)
        # @tensor tmp1[1,2; 4] := AL[loc][1,2,3] * CR[mod1(loc+1, end)][3,4]
        tmp1 = AL[loc] * CR[mod1(loc+1, end)]
        @tensor tmp2[1,3; 4] := CR[loc][1,2] * AR[loc][2,3,4]
        # isapprox(AL[loc] * CR[mod1(loc+1, end)], right, rtol=tol) || return false
        if !isapprox(tmp1, tmp2, rtol=tol)
            err = norm(tmp1 - tmp2)
            println("site $loc is not mixed-canonical, error is $err")
            # println(norm(tmp1), " ", norm(tmp2), " ", dot(tmp1, tmp2))
            return false
        end
    end
    return true
end
ismxiedcanonical(x::MixedCanonicalInfiniteMPS; kwargs...) = ismxiedcanonical(x.AL, x.CR, x.AR; kwargs...)


function mixedcanonicalize(x::InfiniteMPS, alg::InfiniteOrthogonalize)
    alg.normalize || @warn "normalization is enforced"
    AL = x.data.data
    AR = copy(AL)
    CR = [TensorMap(randn, scalartype(x), space_l(item), space_l(item)) for item in AL]
    AL, CR, AR = mixedcanonicalize!(AL, CR, AR, alg)
    return MixedCanonicalInfiniteMPS(AL, CR, AR)
end
function mixedcanonicalize!(x::MixedCanonicalInfiniteMPS, alg::InfiniteOrthogonalize)
    alg.normalize || @warn "normalization is enforced"
    mixedcanonicalize!(x.AL.data, x.CR.data, x.AR.data, alg)
    return x
end
mixedcanonicalize(x::MixedCanonicalInfiniteMPS, alg::InfiniteOrthogonalize) = mixedcanonicalize!(copy(x), alg)

function mixedcanonicalize!(AL, CR, AR, alg::InfiniteOrthogonalize)
    AL, CR = _leftorth!(AL, CR, AR, CR[1], tol=alg.toleig, maxiter=alg.maxitereig)
    AR, CR = _rightorth!(AR, CR, AL, CR[1], tol=alg.toleig, maxiter=alg.maxitereig)
    return AL, CR, AR
end

"""
    solves AL * C = C * A in-place
"""
function _leftorth!(AL, CR, A, C₀; tol=Defaults.tolgauge, maxiter=Defaults.maxiter, eig_miniter::Int=10)
    L = length(A)
    CR[1] = normalize!(C₀)

    iteration = 1
    delta = 2 * tol

    while iteration < maxiter && delta > tol
        if iteration > eig_miniter #when qr starts to fail, start using eigs - should be kw arg
            alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)

            cell = TransferMatrix(AL, A)
            (vals, vecs) = fixedpoint(x -> x * cell, CR[1], :LM, alg)
            (_, CR[1]) = leftorth!(vecs; alg=QRpos())
        end

        cold = CR[1]

        for loc in 1:L
            @tensor tmp[1,3;4] := CR[loc][1,2] * A[loc][2,3,4]
            Q, CR[mod1(loc+1, end)] = leftorth!(tmp, alg=QRpos())
            normalize!(CR[mod1(loc+1, end)])
            AL[loc] = Q
        end

        #update delta
        if domain(cold) == domain(CR[1]) && codomain(cold) == codomain(CR[1])
            delta = norm(cold - CR[1])
        end

        iteration += 1
    end
    # (delta <= tol) && println("leftorth convrged in $iteration iterations")

    delta > tol && @warn "leftorth failed to converge in $(iteration) iterations, error $(delta)"

    return AL, CR
end


function _rightorth!(AR, CR, A, C₀; tol=Defaults.tolgauge, maxiter=Defaults.maxiter, eig_miniter::Int=10)
    L = length(A)
    CR[1] = normalize!(C₀)

    iteration = 1
    delta = 2 * tol
    while iteration < maxiter && delta > tol
        if iteration > eig_miniter#when qr starts to fail, start using eigs
            alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)
            #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

            cell = TransferMatrix(AR, A)
            (vals, vecs) = fixedpoint(x -> cell * x, CR[1], :LM, alg)
            (CR[1], _) = rightorth!(vecs; alg=LQpos())
        end

        cold = CR[1]
        for loc in L:-1:1
            @tensor tmp[1 ; 2 4] := A[loc][1,2,3] * CR[mod1(loc+1, end)][3,4]


            _L, Q = rightorth!(tmp, alg=LQpos())
            AR[loc] = permute(Q, (1,2), (3,))
            CR[loc] = _L
            normalize!(CR[loc])
        end

        #update counters and delta
        if domain(cold) == domain(CR[1]) && codomain(cold) == codomain(CR[1])
            delta = norm(cold - CR[1])
        end

        iteration += 1
    end
    # (delta <= tol) && println("rightorth convrged in $iteration iterations")

    delta > tol && @warn "rightorth failed to converge in $(iteration) iterations, error $(delta)"

    return AR, CR
end

# function _leftorth(A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
#     L = length(A)
#     cell = TransferMatrix(A, A)
#     vl = random_left_boundary(cell)
#     vl = vl * vl'
#     left_eigenvalue, left_eigenvector = largest_eigenpair(x -> x * cell, vl; tol=tol, maxiter=maxiter)
#     C1 = normalize!(chol_split(normalize_trace!(left_eigenvector), tol*10))

#     # C1 = normalize!(sqrt(normalize!(normalize_trace!(left_eigenvector))))
#     CR = Vector{typeof(C1)}(undef, L)
#     _Q1, C1 = leftorth!(C1, alg=QRpos())
#     CR[1] = C1
#     AL = similar.(A)
#     for loc in 1:L
#         @tensor tmp[1,3;4] := CR[loc][1,2] * A[loc][2,3,4]
#         Q, CR[mod1(loc+1, end)] = leftorth!(tmp, alg=QRpos())
#         normalize!(CR[mod1(loc+1, end)])
#         AL[loc] = Q
#     end

#     # # checking left-canonicality
#     # for loc in 1:L
#     #     tmp1 = AL[loc] * CR[mod1(loc+1, end)]
#     #     @tensor tmp2[1,3;4] := CR[loc][1,2] * A[loc][2,3,4]
#     #     normalize!(tmp2)
#     #     isapprox(tmp1, tmp2, rtol=tol*1000) || error("not left-canonical on site $loc")
#     # end
#     return AL, CR, left_eigenvalue
# end

# function _rightorth(A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
#     L = length(A)
#     cell = TransferMatrix(A, A)
#     vr = random_right_boundary(cell)
#     # println("here: ", space(vr))
#     vr = vr * vr'
#     # vr = vr' * vr
#     right_eigenvalue, right_eigenvector = largest_eigenpair(x -> cell * x, vr; tol=tol, maxiter=maxiter)
#     C1 = copy(normalize!(chol_split(normalize_trace!(right_eigenvector), tol*10)'))

#     # C1 = normalize!(copy(sqrt(normalize_trace!(right_eigenvector))'))
#     CR = Vector{typeof(C1)}(undef, L)
#     C1, _ = rightorth!(C1, alg=LQpos())
#     CR[1] = C1
#     AR = similar.(A)
#     for loc in L:-1:1
#         @tensor tmp[1 ; 2 4] := A[loc][1,2,3] * CR[mod1(loc+1, end)][3,4]

#         # U, S, V = stable_tsvd!(tmp, trunc=trunc)
#         # AR[loc] = permute(V, (1,2), (3,))
#         # CR[loc] = U * S
#         # normalize && normalize!(CR[loc])

#         _L, Q = rightorth!(tmp, alg=LQpos())
#         AR[loc] = permute(Q, (1,2), (3,))
#         CR[loc] = _L
#         normalize!(CR[loc])
#     end

#     # # checking right-canonicality
#     # for loc in 1:L
#     #     tmp1 = A[loc] * CR[mod1(loc+1, end)]
#     #     @tensor tmp2[1,3;4] := CR[loc][1,2] * AR[loc][2,3,4]
#     #     isapprox(tmp1, tmp2, rtol=tol*1000) || error("not right-canonical on site $loc")
#     # end

#     return AR, CR, right_eigenvalue
# end


# function get_imps!(AL, CR, AR, trunc::TruncationScheme)
#     U1, S1, V1o = stable_tsvd!(CR[1], trunc=trunc)
#     V1 = V1o
#     svectors = Vector{typeof(S1)}(undef, length(CR))
#     svectors[1] = S1
#     for loc in 2:length(CR)
#         U2, S2, V2 = stable_tsvd!(CR[loc], trunc=trunc)
#         @tensor tmp[1,3;5] := V1[1,2] * AR[loc-1][2,3,4] * V2'[4,5]
#         svectors[loc] = S2
#         AR[loc-1] = tmp
#         V1 = V2
#     end
#     @tensor tmp[1,3;5] := V1[1,2] * AR[end][2,3,4] * V1o'[4,5]
#     AR[end] = tmp
#     return InfiniteMPS(AR, svectors)
# end



######## algorithm from MPSKit.jl #########

# """
#     solves AL * C = C * A in-place
# """
# function uniform_leftorth!(AL, CR, A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
#     #(_,CR[end]) = leftorth!(CR[end], alg = TensorKit.QRpos());
#     normalize!(CR[end])

#     iteration = 1
#     delta = 2 * tol

#     while iteration < maxiter && delta > tol
#         if iteration > 10 #when qr starts to fail, start using eigs - should be kw arg
#             alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)

#             (vals, vecs) = eigsolve(flip(TransferMatrix(A, AL)), CR[end], 1, :LM, alg)
#             (_, CR[end]) = leftorth!(vecs[1]; alg=TensorKit.QRpos())
#         end

#         cold = CR[end]

#         for loc in 1:length(AL)
#             AL[loc] = _transpose_front(CR[mod1(loc - 1, end)] * _transpose_tail(A[loc]))
#             AL[loc], CR[loc] = leftorth!(AL[loc]; alg=QRpos())
#             normalize!(CR[loc])
#         end

#         #update delta
#         if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
#             delta = norm(cold - CR[end])
#         end

#         iteration += 1
#     end

#     delta > tol && @warn "leftorth failed to converge $(delta)"

#     return AL, CR
# end

# """
#     solves C * AR = C * A in-place
# """
# function uniform_rightorth!(AR, CR, A; tol=Defaults.tolgauge, maxiter=Defaults.maxiter)
#     #(CR[end],_) = rightorth!(CR[end], alg = TensorKit.LQpos());
#     normalize!(CR[end])

#     iteration = 1
#     delta = 2 * tol
#     while iteration < maxiter && delta > tol
#         if iteration > 10#when qr starts to fail, start using eigs
#             alg = Arnoldi(; krylovdim=30, tol=max(delta * delta, tol / 10), maxiter=maxiter)
#             #Projection of the current guess onto its largest self consistent eigenvector + isolation of the unitary part

#             (vals, vecs) = eigsolve(TransferMatrix(A, AR), CR[end], 1, :LM, alg)
#             (CR[end], _) = rightorth!(vecs[1]; alg=TensorKit.LQpos())
#         end

#         cold = CR[end]
#         for loc in length(AR):-1:1
#             AR[loc] = A[loc] * CR[loc]

#             CR[mod1(loc - 1, end)], temp = rightorth!(_transpose_tail(AR[loc]); alg=LQpos())
#             AR[loc] = _transpose_front(temp)
#             normalize!(CR[mod1(loc - 1, end)])
#         end

#         #update counters and delta
#         if domain(cold) == domain(CR[end]) && codomain(cold) == codomain(CR[end])
#             delta = norm(cold - CR[end])
#         end

#         iteration += 1
#     end

#     delta > tol && @warn "rightorth failed to converge $(delta)"

#     return AR, CR
# end
