"""
    Sampler(y::Vector{Int}, X::Matrix{Float64}; kwargs...)

Initialize a Gibbs Sampler for the following Bayesian Negative-Binomial model:

```math
\\begin{aligned}
y_i | x_i, \\beta
&\\sim 
\\text{NegativeBinomial}(r_0, \\Lambda(x_i'\\beta)),
\\\\
\\Lambda(x_i'\\beta)
&=
1 / (1 + e^{- x_i'\\beta}),
\\\\
\\beta 
&\\sim 
\\mathcal{N}(m_0, \\Sigma_0),
\\end{aligned}
```

given a response vector `y` and a design matrix `X`. This constructor does not 
copy any of its arguments. Hence, for example, if `y[i]` is changed, it will 
affect the Gibbs sampler.

# Keyword arguments

* `β = zeros(size(X, 2))`: current state of ``\\beta``.
* `Σ0 = 10 * I(size(X, 2))`: ``\\Sigma_0``.
* `m0 = zeros(size(X, 2))`: ``m_0``.
* `r0 = [1]`: ``[r_0]``.

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
    β::Vector{Float64}
    w::Vector{Float64}
    z::Vector{Float64}
    a::Vector{Float64}
    m0::Vector{Float64}
    Σ0::Matrix{Float64}
    r0::Vector{Int}
    function Sampler(
        y::Vector{Int}, 
        X::Matrix{Float64};
        β::Vector{Float64} = zeros(size(X, 2)),
        m0::Vector{Float64} = zeros(size(X, 2)), 
        Σ0::Matrix{Float64} = Matrix{Float64}(10 * I(size(X, 2))),
        r0::Vector{Int} = [1],
    )
        N, D = size(X, 1)
        w = zeros(N)
        z = zeros(N)
        a = zeros(D)
        new(y, X, β, w, z, a, m0, Σ0, r0)
    end
end

"""
    step!(rng::AbstractRNG, mdl::BayesNegativeBinomial.Sampler)

Perform 1 Gibbs iteration, using the data augmentation strategy 
described in Polson et al. (2013).

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
function step!(rng::AbstractRNG, s::Sampler)
    @extract s : y X β w z a m0 Σ0 r0
    mul!(z, X, β);
    for i in 1:length(w)
        w[i] = rand(rng, PolyaGammaPSWSampler(y[i] + r0[], z[i]))
    end
    Σ1 = inv(cholesky(Symmetric(X' * Diagonal(w) * X + inv(Σ0))))
    m1 = Σ1 * (X' * (y .- r0[]) / 2 + Σ0 \ m0)
    rand!(rng, MvNormal(m1, Σ1), β)
end

"""
    sample(rng::AbstractRNG, s::BayesNegativeBinomial.Sampler; kwargs...)

Draw a posterior sample using the Gibbs sampler `s`, following the data 
augmentation strategy described in Polson et al. (2013).

# Arguments

* `mcmcsize = 2000`: the total number of MCMC iterations.
* `burnin = 1000`: the number of *burn-in* iterations.

# Example 

```julia
julia> using Random                 
julia> rng = MersenneTwister(1)     
julia> X = randn(rng, 100, 2)
julia> y = rand(rng, 0:2, 10)        
julia> s = BayesNegativeBinomial.Sampler(y, X)        
julia> chain = BayesNegativeBinomial.sample(rng, s)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.    
"""
function sample(rng::AbstractRNG, s::Sampler; mcmcsize = 4000, burnin = 2000)
    chain = [zeros(size(s.X, 2)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, s)
        if iter > burnin
            chain[iter - burnin] .= mdl.β
        end
    end
    return chain
end
