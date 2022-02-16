using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = - 0.5 * ones(D)
    X = rand(N, D)
    g = ones(Bool, 3)
    g[2] = false
    mapping = [[1], collect(2:D-1), [D]]
    gexp = zeros(Bool, D)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end
    p = 1 ./ (1 .+ exp.(- X[:, gexp] * β[gexp]))
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
