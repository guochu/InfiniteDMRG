push!(LOAD_PATH, dirname(dirname(Base.@__DIR__)) * "/DMRG/src")


include(dirname(Base.@__DIR__) * "/src/includes.jl")


using Test
using TestExtras

include("states.jl")