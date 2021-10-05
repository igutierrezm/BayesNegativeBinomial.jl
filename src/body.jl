"""
    Sampler(y::Vector{Int}, X::Matrix{Float64}; kwargs...)

Initialize a Gibbs Sampler for the following Bayesian Negative-Binomial model:

```math
\\begin{aligned}
y_i | x_i, \\beta
&\\sim 
\\text{NegativeBinomial}(r_{0y}, \\Lambda(x_i'\\beta)),
\\\\
\\Lambda(x_i'\\beta)
&=
1 / (1 + e^{- x_i'\\beta}),
\\\\
\\beta 
&\\sim 
\\mathcal{N}(m_{0\\beta}, \\Sigma_{0\\beta}),
\\end{aligned}
```

given a response vector `y` and a design matrix `X`. This constructor does not 
copy any of its arguments. Hence, for example, if `y[1]` is changed, it will 
affect the Gibbs sampler.

# Keyword arguments

* `β = zeros(size(X, 2))`: current state of ``\\beta``.
* `Σ0β = 10 * I(size(X, 2))`: ``\\Sigma_{0\\beta}``.
* `m0β = zeros(size(X, 2))`: ``m_{0\\beta}``.
* `r0y = [1]`: ``[r_{0y}]``.

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
    γ::Vector{Bool}
    m0β::Vector{Float64}
    Σ0β::Matrix{Float64}
    r0y::Vector{Int}
    function Sampler(
        y::Vector{Int}, 
        X::Matrix{Float64};
        β::Vector{Float64} = zeros(size(X, 2)),
        m0β::Vector{Float64} = zeros(size(X, 2)), 
        Σ0β::Matrix{Float64} = Matrix{Float64}(10 * I(size(X, 2))),
        r0y::Vector{Int} = [1],
    )
        N, D = size(X)
        w = zeros(N)
        z = zeros(N)
        a = zeros(D)
        γ = ones(Bool, D)
        new(y, X, β, w, z, a, γ, m0β, Σ0β, r0y)
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
julia> s = BayesNegativeBinomial.Sampler(y, X)        
julia> chain = BayesNegativeBinomial.sample(rng, s)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.    
"""
function sample(rng::AbstractRNG, s::Sampler; mcmcsize = 10000, burnin = 5000)
    chain = [zeros(size(s.X, 2)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(rng, s)
        if iter > burnin
            chain[iter - burnin] .= s.β
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
function step!(rng::AbstractRNG, s::Sampler)
    @extract s : y X β w z a γ m0β Σ0β r0y
    # Update w
    mul!(z, X, β);
    for i in 1:length(w)
        w[i] = rand(rng, PolyaGammaPSWSampler(y[i] + r0y[], z[i]))
    end

    # Update some auxiliary statistics
    A = Symmetric(X' * Diagonal(w) * X + inv(Σ0β))
    b = X' * (y .- r0y[]) / 2

    # Update gamma
    for d in 1:length(γ)
        # Logodds numerator
        γ[d] = false
        mf, Σf = posterior_hyperparameters(s::Sampler, A, b)
        logodds_den = 
            logpdf(MvNormal(m0β[γ], Σ0β[γ, γ]), zeros(length(mf))) -
            logpdf(MvNormal(mf, Σf), zeros(length(mf)))
        # Logodds numerator
        γ[d] = true
        mt, Σt = posterior_hyperparameters(s::Sampler, A, b)
        logodds_num = 
            logpdf(MvNormal(m0β[γ], Σ0β[γ, γ]), zeros(length(mt))) -
            logpdf(MvNormal(mt, Σt), zeros(length(mt)))
        odds = exp(logodds_num - logodds_den)
        γ[d] = rand(rng) < odds / (1.0 + odds)
    end

    # Update the posterior hyperparameters 
    β .= 0.0
    m1, Σ1 = posterior_hyperparameters(s::Sampler, A, b)
    β[γ] .= rand(rng, MvNormal(m1, Σ1))
    return nothing
end

function posterior_hyperparameters(s::Sampler, A, b)
    @extract s : γ m0β Σ0β
    Σ1 = inv(cholesky(A[γ, γ]))
    m1 = Σ1 * (b[γ] + Σ0β[γ, γ] \ m0β[γ])
    return m1, Σ1
end

