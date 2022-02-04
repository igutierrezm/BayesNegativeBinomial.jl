using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = - 0.5 * ones(D)
    X = rand(N, D)
    γ = ones(Bool, 3)
    γ[2] = false
    mapping = [[1], collect(2:D-1), [D]]
    γexp = zeros(Bool, D)
    for d in 1:length(mapping)
        γexp[mapping[d]] .= γ[d]
    end
    p = 1 ./ (1 .+ exp.(- X[:, γexp] * β[γexp]))
    y = [rand(rng, NegativeBinomial(2, 1 - p[i])) for i in 1:N]
    return p, y, X, mapping
end

@testset "BayesNegativeBinomial.jl" begin
    N, D = 2000, 4
    rng = MersenneTwister(1)
    p, y, X, mapping = simulate_sample(rng, N, D)
    sampler = BayesNegativeBinomial.Sampler(y, X; mapping)
    BayesNegativeBinomial.step!(rng, sampler)
    sampler = BayesNegativeBinomial.Sampler(y, X)
    chain = BayesNegativeBinomial.sample(rng, sampler)
    mean(chain)
    chain_array = hcat(chain...)
    println(mean(chain_array .== 0.0, dims = 2))
end
