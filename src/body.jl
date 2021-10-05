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
    \\mathcal{N}(m_{0\\beta}, \\Sigma_{0\\beta}), 
    &\\text{if } \\gamma = 1,
    \\\\
    \\delta_0
    &\\text{if } \\gamma = 0,
\\end{cases}
\\\\
\\gamma
&\\sim
\\text{Womack}(\\eta_{0\\gamma}),
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
    A::Matrix{Float64}
    b::Vector{Float64}
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
        A = zeros(D, D)
        b = zeros(D)
        new(y, X, β, w, z, a, γ, A, b, m0β, Σ0β, r0y)
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
    step_w!(rng, s)
    step_γ!(rng, s)
    step_β!(rng, s)
    return nothing
end

function step_w!(rng::AbstractRNG, s::Sampler)
    @extract s : y X β w z r0y
    mul!(z, X, β);
    for i in 1:length(w)
        w[i] = rand(rng, PolyaGammaPSWSampler(y[i] + r0y[], z[i]))
    end    
    return nothing
end

function step_γ!(rng::AbstractRNG, s::Sampler)
    @extract s : A b γ m0β Σ0β
    step_A!(s)
    step_b!(s)
    for d in 1:length(γ)
        logodds = 0.0
        for val in 0:1
            γ[d] = val
            m1, Σ1 = posterior_hyperparameters(s)
            logodds += (-1)^(val + 1) * (
                logpdf(MvNormal(m0β[γ], Σ0β[γ, γ]), zeros(length(m1))) -
                logpdf(MvNormal(m1, Σ1), zeros(length(m1)))
            )
        end
        γ[d] = rand(rng) < exp(logodds) / (1.0 + exp(logodds))
    end
    return nothing
end

function step_β!(rng::AbstractRNG, s::Sampler)
    @extract s : β A b γ
    β .= 0.0
    m1, Σ1 = posterior_hyperparameters(s)
    β[γ] .= rand(rng, MvNormal(m1, Σ1))
    return nothing
end

function posterior_hyperparameters(s::Sampler)
    @extract s : γ A b m0β Σ0β
    Σ1 = inv(cholesky(Symmetric(A[γ, γ])))
    m1 = Σ1 * (b[γ] + Σ0β[γ, γ] \ m0β[γ])
    return m1, Σ1
end

function step_A!(s::Sampler)
    @extract s : X w Σ0β A
    A .= X' * Diagonal(w) * X + inv(Σ0β)
    return nothing
end

function step_b!(s::Sampler)
    @extract s : X y r0y b
    b .= X' * (y .- r0y[]) / 2
    return nothing
end
