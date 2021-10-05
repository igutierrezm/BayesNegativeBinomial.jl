using Revise
using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = ones(D)
    X = rand(N, D)
    p = 1 ./ (1 .+ exp.(- X[:, 1:(D-1)] * β[1:(D-1)]))
    y = [rand(rng, NegativeBinomial(1, p[i])) for i in 1:N]
    return y, X
end

@testset "BayesNegativeBinomial.jl" begin
    N, D = 1000, 4
    rng = MersenneTwister(1)
    y, X = simulate_sample(rng, N, D)
    s = BayesNegativeBinomial.Sampler(y, X)
    BayesNegativeBinomial.step!(rng, s)
    chain = BayesNegativeBinomial.sample(rng, s)
    chain_array = hcat(chain...) 
    mean(chain_array .== 0.0, dims = 2)
end

# mean(chain)