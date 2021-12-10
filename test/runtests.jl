using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = - 0.5 * ones(D)
    X = rand(N, D)
    γ = ones(Bool, D)
    γ[2] = false
    γ[4] = false
    p = 1 ./ (1 .+ exp.(- X[:, γ] * β[γ]))
    y = [rand(rng, NegativeBinomial(2, 1 - p[i])) for i in 1:N]
    return p, y, X
end

@testset "BayesNegativeBinomial.jl" begin
    N, D = 2000, 4
    rng = MersenneTwister(1)
    p, y, X = simulate_sample(rng, N, D)
    sampler = BayesNegativeBinomial.Sampler(y, X)
    BayesNegativeBinomial.step!(rng, sampler)
    sampler = BayesNegativeBinomial.Sampler(y, X)
    chain = BayesNegativeBinomial.sample(rng, sampler)
    mean(chain)
    chain_array = hcat(chain...)
    println(mean(chain_array .== 0.0, dims = 2))
end
