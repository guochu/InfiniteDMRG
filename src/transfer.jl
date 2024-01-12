# function transfer_matrix(A::MPSTensor, B::MPSTensor)
# 	@tensor tmp[1,4; 3,5] := conj(A[1,2,3]) * B[4,2,5]
# end

# function transfer_matrix(A::MPOTensor, B::MPOTensor)
# 	@tensor tmp[1,5;3,6] := conj(A[1,2,3,4]) * B[5,2,6,4]
# end

# function transfer_matrix(A::MPSTensor, m::MPOTensor, B::MPSTensor)
# 	@tensor tmp[1,4,7;8,5,3] := (conj(A[1,2,3]) * m[4,2,5,6]) * B[7,6,8]
# end

# function transfer_matrix(A::MPOTensor, m::MPOTensor, B::MPOTensor)
# 	@tensor tmp[1,5,8;9,6,3] := (conj(A[1,2,3,4]) * m[5,2,6,7]) * B[8,7,9,4]
# end

# function updatecyclicleft(hold::MPOTensor, A::MPSTensor, B::MPSTensor)
# 	@tensor tmp[1,2;6,7] := (hold[1,2,3,4] * conj(A[3,5,6])) * B[4,5,7]
# end

# function updatecyclicleft(hold::MPOTensor, A::MPOTensor, B::MPOTensor)
# 	@tensor tmp[1,2;6,8] := (hold[1,2,3,4] * conj(A[3,5,6,7])) * B[4,5,8,7]
# end

# function transfer_left(left::MPSBondTensor, a::MPSTensor, b::MPSTensor)
# 	@tensor tmp[3;5] := conj(a[1,2,3]) * left[1,4] * b[4,2,5]
# end
# function transfer_left(left::MPSBondTensor, a::MPOTensor, b::MPOTensor)
# 	@tensor tmp[3;6] := conj(a[1,2,3,4]) * left[1,5] * b[5,2,6,4]
# end
# function transfer_right(right::MPSBondTensor, a::MPSTensor, b::MPSTensor)
# 	@tensor tmp[4;5] := right[1, 2] * b[4,3,1] * conj(a[5,3,2])
# end
# function transfer_right(right::MPSBondTensor, a::MPOTensor, b::MPOTensor)
# 	@tensor tmp[4;6] := right[1, 2] * b[4,3,1,5] * conj(a[6,3,2,5])
# end


struct TransferMatrix{M <: Union{MPSTensor, MPOTensor}}
	above::Vector{M}
	below::Vector{M}
end

Base.length(x::TransferMatrix) = length(x.above)
TK.scalartype(::Type{TransferMatrix{M}}) where M = scalartype(M)

function transfer_left(left::MPSBondTensor, m::TransferMatrix)
	for (a, b) in zip(m.above, m.below)
		left = updateleft(left, a, b)
	end
	return left
end

function transfer_right(right::MPSBondTensor, m::TransferMatrix)
	for (a, b) in Iterators.reverse(zip(m.above, m.below))
		right = updateright(right, a, b)
	end
	return right
end

random_left_boundary(x::TransferMatrix) = TensorMap(randn, scalartype(x), space_l(x.above[1]), space_l(x.below[1]))
random_right_boundary(x::TransferMatrix) = TensorMap(randn, scalartype(x), space_r(x.below[end])', space_r(x.above[end])')

random_boundaries(x::TransferMatrix) = (random_left_boundary(x), random_right_boundary(x))

# function Base.convert(::Type{<:TensorMap}, x::TransferMatrix)
# 	hleft = transfer_matrix(x.above[1], x.below[1])
# 	for i in 2:length(x.above)
# 		hleft = updatecyclicleft(hleft, x.above[i], x.below[i])
# 	end
# 	return hleft
# end

# struct ExpTransferMatrix{M <: AbstractSparseMPOTensor, V <: Union{MPSTensor, MPOTensor}}
# 	above::Vector{V}
# 	middle::Vector{M}
# 	below::Vector{V}
# end

# Base.length(x::ExpTransferMatrix) = length(x.above)
# TK.scalartype(::Type{ExpTransferMatrix{M, V}}) where {M, V} = promote_type(scalartype(M), scalartype(V))

# function transfer_left(left::Vector{<:MPSBondTensor}, m::ExpTransferMatrix) 
# 	for (a, h, b) in Iterators.reverse(zip(m.above, m.middle, m.below))
# 		left = updateleft(left, a, h, b)
# 	end	
# 	return left
# end

# function transfer_right(right::Vector{<:MPSBondTensor}, m::ExpTransferMatrix)
# 	for (a, h, b) in Iterators.reverse(zip(m.above, m.middle, m.below))
# 		right = updateright(right, a, h, b)
# 	end
# 	return right
# end

