# using BayesNegativeBinomial
# using Distributions
# using LinearAlgebra
# using Random 
# using Test
# const BNB = BayesNegativeBinomial

# function simulate_sample(rng, N, D)
#     X = [ones(N) rand(N, D - 1)]
#     β = [-6.0; 10.0 * ones(D - 1)]
#     g = [true, false, true]
#     mapping = [[1], collect(2:D-1), [D]]
#     gexp = zeros(Bool, D)
#     for d in 1:length(mapping)
#         gexp[mapping[d]] .= g[d]
#     end
#     p = 1 ./ (1 .+ exp.(X[:, gexp] * β[gexp]))
#     y = [rand(rng, NegativeBinomial(2, p[i])) for i in 1:N]
#     return p, y, X, mapping
# end

# begin
#     N, D = 2000, 5
#     rng = MersenneTwister(1)
#     p, y, X, mapping = simulate_sample(rng, N, D)
#     println(mean(y))
#     sampler = BayesNegativeBinomial.Sampler(y, X; mapping, Σ0β = Matrix(1000.0 * I(D)), update_g = [false, true, false])
#     chain = BayesNegativeBinomial.sample(rng, sampler, mcmcsize = 10000, burnin = 5000)
#     mean(chain)
# end
# # @testset "BayesNegativeBinomial.jl" begin
# #     N, D = 2000, 4
# #     rng = MersenneTwister(1)
# #     p, y, X, mapping = simulate_sample(rng, N, D)
# #     sampler = BayesNegativeBinomial.Sampler(y, X; mapping)
# #     BayesNegativeBinomial.step!(rng, sampler)
# #     sampler = BayesNegativeBinomial.Sampler(y, X)
# #     chain = BayesNegativeBinomial.sample(rng, sampler)
# #     mean(chain)
# #     chain_array = hcat(chain...)
# #     println(mean(chain_array .== 0.0, dims = 2))
# # end
