push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")


include(dirname(Base.@__DIR__) * "/src/includes.jl")


using Test
using TestExtras

include("util.jl")

include("states.jl")
include("mpo.jl")


### algorithms
include("algorithms/expectation.jl")
include("algorithms/timeevompo.jl")
include("algorithms/approximate.jl")