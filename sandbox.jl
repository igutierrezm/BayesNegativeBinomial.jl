using Distributions
using ExtractMacro
using PolyaGammaSamplers
using LinearAlgebra
using Random

function simulate_sample(rng, N, D)
    β = ones(D)
    X = rand(N, D)
    p = 1 ./ (1 .+ exp.(- X * β))
    y = [rand(rng, NegativeBinomial(1, p[i])) for i in 1:N]
    return y, X
end

function foo(rng, y, X, r, m0β, Σ0β)
    N, D = size(X)
    w = zeros(N)
    z = zeros(N)
    β = zeros(D)
    a = X' * (y .- r) / 2 + Σ0β \ m0β
    mcmcsize = 4000
    burnin = 2000
    chain = [zeros(D) for i in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        mul!(z, X, β);
        for i in 1:length(w)
            w[i] = rand(rng, PolyaGammaPSWSampler(y[i] + r, z[i]))
        end
        Σ1β = inv(cholesky(Symmetric(X' * Diagonal(w) * X + inv(Σ0β))))
        rand!(rng, MvNormal(Σ1β * a, Σ1β), β)            
        if iter > burnin
            chain[iter - burnin] .= β
        end
    end
    return chain
end

m0β = zeros(D)
Σ0β = 100 * I(D)
N, D = 2000, 4;
rng = MersenneTwister(1);
y, X = simulate_sample(rng, N, D);
maximum(y)
chain = foo(rng, y, X, 1, m0β, Σ0β);
mean(chain)
