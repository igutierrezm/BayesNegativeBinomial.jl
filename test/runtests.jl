using BayesNegativeBinomial
using Distributions
using Random 
using Test

function simulate_sample(rng, N, D)
    β = ones(D)
    X = rand(N, D)
    p = 1 ./ (1 .+ exp.(- X * β))
    y = [rand(rng, NegativeBinomial(1, p[i])) for i in 1:N]
    return y, X
end

@testset "BayesNegativeBinomial.jl" begin
    N, D = 4000, 4;
    rng = MersenneTwister(1);
    y, X = simulate_sample(rng, N, D);
    # s = BayesNegativeBinomial.Sampler(y, X)
    # chain = BayesNegativeBinomial.sample(rng, s);
end
