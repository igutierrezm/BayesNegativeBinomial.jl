using Revise
using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = ones(D)
    X = rand(N, D)
    # γ = [false, true, false, false]
    γ = ones(Bool, D)
    γ[2] = false
    γ[4] = false
    p = 1 ./ (1 .+ exp.(- X[:, γ] * β[γ]))
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