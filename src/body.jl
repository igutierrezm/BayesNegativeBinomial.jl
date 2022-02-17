"""
    Sampler(y::Vector{Int}, X::Matrix{Float64}; kwargs...)

Initialize a Gibbs Sampler for the following Bayesian Negative-Binomial model:

```math
\\begin{aligned}
y_i | x_i, \\beta, \\gamma
&\\sim 
\\text{NB}(r_{0y}, 1 / (1 + e^{- x_{i\\gamma}'\\beta_{\\gamma}})),
\\\\
\\beta_k | \\gamma_k
&\\sim 
\\begin{cases}
    \\mathcal{N}(\\mu_{0\\beta}, \\Sigma_{0\\beta}), 
    &\\text{if } \\gamma = 1,
    \\\\
    \\delta_0
    &\\text{if } \\gamma = 0,
\\end{cases}
\\\\
\\gamma
&\\sim
\\text{Womack}(\\eta_{0\\gamma}),
\\\\
s
&\\sim
\\text{Gamma}(a_{0s}, b_{0s}),
\\end{aligned}
```

given a response vector `y` and a design matrix `X`. This constructor does not 
copy any of its arguments. Hence, for example, if `y[1]` is changed, it will 
affect the Gibbs sampler.

where ``y_i \\in \\mathbb{N}_0`` is the response for the ``i``-observation, 
``x_i \\in \\mathbb{R}^K`` is the associated covariate vector, 
``\\delta_0(\\cdot)`` is the Dirac measure centred at ``0`` and 
``\\text{Womack}(\\eta_0)`` is the Womack distribution with 
parameter ``\\eta_{0\\gamma}`` on ``\\{0, 1\\}^K``.

# Keyword arguments

* `β = zeros(size(X, 2))`: current state of ``\\beta``.
* `Σ0β = 10 * I(size(X, 2))`: ``\\Sigma_{0\\beta}``.
* `μ0β = zeros(size(X, 2))`: ``\\mu_{0\\beta}``.
* `a0s = 1.0`: ``a_{0s}``.
* `b0s = 1.0`: ``a_{0s}``.

# Example 

```julia
julia> using Random, Distributions
julia> rng = MersenneTwister(1)
julia> X = randn(rng, 100, 2)
julia> y = rand(rng, 0:2, 10)
julia> s = BayesNegativeBinomial.Sampler(y, X)
```
"""
struct Sampler
    y::Vector{Int}
    X::Matrix{Float64}
    mapping::Vector{Vector{Int}}
    β::Vector{Float64}
    ω::Vector{Float64}
    ξ::Vector{Float64}
    ϕ::Vector{Float64}
    q::Vector{Float64}
    g::Vector{Bool}
    A::Matrix{Float64}
    b::Vector{Float64}
    s::Vector{Int}
    a0s::Float64
    b0s::Float64
    μ0β::Vector{Float64}
    Σ0β::Matrix{Float64}
    ζ0g::Float64
    update_g::Vector{Bool}
    function Sampler(
        y::Vector{Int}, 
        X::Matrix{Float64};
        β::Vector{Float64} = zeros(size(X, 2)),
        s::Vector{Int} = [2],
        μ0β::Vector{Float64} = zeros(size(X, 2)), 
        Σ0β::Matrix{Float64} = Matrix{Float64}(10 * I(size(X, 2))),
        mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X, 2)],
        a0s::Float64 = 1.0,
        b0s::Float64 = 1.0,
        ζ0g::Float64 = 1.0,
        update_g = ones(Bool, length(mapping))
    )
        N, D = size(X)
        ω = zeros(N)
        ξ = zeros(N)
        ϕ = zeros(N)
        q = zeros(N)
        g = ones(Bool, length(mapping))
        A = zeros(D, D)
        b = zeros(D)
        new(y, X, mapping, β, ω, ξ, ϕ, q, g, A, b, s, a0s, b0s, μ0β, Σ0β, ζ0g, update_g)
    end
end

"""
    sample(rng::AbstractRNG, s::BayesNegativeBinomial.Sampler; kwargs...)

Draw a posterior sample using the Gibbs sampler `s`, 
following Polson et al. (2013).    

# Arguments

* `mcmcsize = 2000`: the total number of MCMC iterations.
* `burnin = 1000`: the number of *burn-in* iterations.

# Example 

```julia
julia> using Random                 
julia> rng = MersenneTwister(1)     
julia> X = randn(rng, 100, 2)
julia> y = rand(rng, 0:2, 10)        
julia> sampler = BayesNegativeBinomial.Sampler(y, X)        
julia> chain = BayesNegativeBinomial.sample(rng, sampler)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.    
"""
function sample(rng::AbstractRNG, sampler::Sampler; mcmcsize = 10000, burnin = 5000)
    chain = [zeros(size(sampler.X, 2)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, sampler)
        if iter > burnin
            chain[iter - burnin] .= sampler.β
        end
    end
    return chain
end

"""
    step!(rng::AbstractRNG, s::BayesNegativeBinomial.Sampler)

Perform 1 iteration of the Gibbs sampler `s`, following Polson et al. (2013).

# Example 

```julia
julia> using Random, Distributions
julia> rng = MersenneTwister(1)
julia> X = randn(rng, 100, 2)
julia> y = rand(rng, 0:2, 10)
julia> s = BayesNegativeBinomial.Sampler(y, X)
julia> BayesNegativeBinomial.step!(rng, s)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.
"""
function step!(rng::AbstractRNG, sampler::Sampler)
    step_ϕ!(sampler)
    step_ξ!(sampler)
    # step_s!(rng, sampler)
    step_ω!(rng, sampler)
    step_A!(sampler)
    step_b!(sampler)    
    step_g!(rng, sampler)
    step_β!(rng, sampler)
    return nothing
end

function step_ξ!(sampler::Sampler)
    (; ξ, X, β) = sampler
    mul!(ξ, X, β)
    return nothing
end

function step_ϕ!(sampler::Sampler)
    (; ξ, ϕ) = sampler
    @. ϕ = 1.0 / (1.0 + exp(ξ))
    return nothing
end

function step_ω!(rng::AbstractRNG, sampler::Sampler)
    (; y, ω, ξ, s) = sampler
    for i in 1:length(ω)
        ω[i] = rand(rng, PolyaGammaPSWSampler(y[i] + s[], ξ[i]))
    end    
    return nothing
end

function step_g!(rng::AbstractRNG, sampler::Sampler)
    (; update_g, mapping, g, μ0β, Σ0β, ζ0g) = sampler
    D = length(μ0β)
    gexp = zeros(Bool, D)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end
    pg = Womack(length(g), ζ0g)
    for d in 1:length(g)
        update_g[d] || continue        
        logodds = 0.0
        for val in 0:1
            g[d] = val
            gexp[mapping[d]] .= val
            m1, Σ1 = posterior_hyperparameters(sampler)
            logodds += (-1)^(val + 1) * (
                logpdf(pg, g) +
                logpdf(MvNormal(μ0β[gexp], Σ0β[gexp, gexp]), zeros(sum(gexp))) -
                logpdf(MvNormal(m1, Σ1), zeros(length(m1)))
            )
        end
        g[d] = rand(rng) < exp(logodds) / (1.0 + exp(logodds))
        gexp[mapping[d]] .= g[d]
    end
    return nothing
end

function step_β!(rng::AbstractRNG, sampler::Sampler)
    (; mapping, β, g) = sampler
    D = length(β)
    gexp = zeros(Bool, D)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end        
    β .= 0.0
    m1, Σ1 = posterior_hyperparameters(sampler)
    β[gexp] .= rand(rng, MvNormal(m1, Σ1))
    return nothing
end

function posterior_hyperparameters(sampler::Sampler)
    (; mapping, g, A, b, μ0β, Σ0β) = sampler
    D = length(μ0β)
    gexp = zeros(Bool, D)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end    
    Σ1 = inv(cholesky(Symmetric(A[gexp, gexp])))
    m1 = Σ1 * (b[gexp] + Σ0β[gexp, gexp] \ μ0β[gexp])
    return m1, Σ1
end

function step_A!(sampler::Sampler)
    (; X, ω, Σ0β, A) = sampler
    A .= X' * Diagonal(ω) * X + inv(Σ0β)
    return nothing
end

function step_b!(sampler::Sampler)
    (; X, y, s, b) = sampler
    b .= X' * (y .- s[]) / 2
    return nothing
end

# function step_s!(rng, sampler::Sampler)
#     (; y, q, ϕ, s, a0s, b0s) = sampler
#     N = length(q)
#     s0 = s[]
#     for i in 1:N
#         q[i] = 0
#         for j in 1:y[i]
#             q[i] += rand(rng) <= s0 / (s0 + j - 1)
#         end
#     end
#     a1s = a0s + sum(q)
#     b1s = b0s
#     for i in 1:N
#         b1s - log(1 - ϕ[i]) 
#     end
#     ds = Gamma(a1s, 1 / b1s)
#     s[] = rand(rng, ds)
#     return nothing
# end
  
struct Womack <: DiscreteMultivariateDistribution
    D::Int
    ζ::Float64
    p::Vector{Float64}
    punnormalized::Vector{Float64}
    function Womack(D::Int, ζ::Float64)
        p = big.([zeros(D); 1.0])
        for d1 in (D - 1):-1:0
            for d2 in 1:(D - d1)
                p[1 + d1] += ζ * p[1 + d1 + d2] * binomial(big(d1 + d2), big(d1))
            end
        end
        p /= sum(p)
        punnormalized = copy(p)
        for d1 in 1:D
            p[d1] /= binomial(big(D), big(d1 - 1))
        end
        return new(D, ζ, p, punnormalized)
    end
end

function pdf(d::Womack, g::Vector{Bool})
    return d.p[sum(g) + 1]
end

function logpdf(d::Womack, g::Vector{Bool})
    return log(pdf(d, g))
end