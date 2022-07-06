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
julia> X = randn(100, 2)
julia> y = rand(0:2, 10)
julia> s = BayesNegativeBinomial.Sampler(y, X)
```
"""
struct Sampler
    # Data
    y::Vector{Int}
    X::Matrix{Float64}
    mapping::Vector{Vector{Int}}
    update_g::Vector{Bool}
    update_s::Bool
    # Hyperparameters
    ζ0g::Float64
    a0s::Float64
    b0s::Float64
    μ0β::Vector{Float64}
    Σ0β::Matrix{Float64}
    # Parameters
    g::Vector{Bool}
    β::Vector{Float64}
    ω::Vector{Float64}
    q::Vector{Float64}
    s::Vector{Int}
    # Transformed Data
    N::Int
    D::Int
    # Transformed Parameters
    gexp::Vector{Bool} # g[mapping]
    ξ::Vector{Float64} # ξ := exp(X * β)
    ϕ::Vector{Float64} # ϕ := 1.0 / (1.0 + exp(ξ))
    A::Matrix{Float64} # A := X' * Diagonal(ω) * X + inv(Σ0β)
    b::Vector{Float64} # b := X' * (y .- s[]) / 2
    function Sampler(
        # Data
        y::Vector{Int}, 
        X::Matrix{Float64};
        g::Vector{Float64} = ones(size(X, 2)),
        mapping::Vector{Vector{Int}} = [[i] for i in 1:size(X, 2)],
        update_g::Bool = false,
        update_s::Bool = false,
        # Hyperparameters
        ζ0g::Float64 = 1.0,
        a0s::Float64 = 1.0,
        b0s::Float64 = 1.0,
        μ0β::Vector{Float64} = zeros(size(X, 2)), 
        Σ0β::Matrix{Float64} = Matrix{Float64}(I(size(X, 2))),
        # Parameters
        β::Vector{Float64} = zeros(size(X, 2)),
        ω::Vector{Float64} = zeros(size(X, 1)),
        q::Vector{Float64} = zeros(size(X, 1)),
        s::Vector{Int} = [2],
    )
        # Transformed Data
        N, D = size(X)
        # Transformed Parameters
        gexp = zeros(Bool, D)
        for d in 1:length(mapping)
            gexp[mapping[d]] .= g[d]
        end
        ξ = zeros(N)
        ϕ = zeros(N)
        A = zeros(D, D)
        b = zeros(D)
        # Final struct
        new(
            y, X, mapping, update_g, update_s, # data
            ζ0g, a0s, b0s, μ0β, Σ0β,           # hyperparameters
            g, β, ω, q, s,                     # parameters 
            N, D,                              # transformed data
            gexp, ξ, ϕ, A, b                   # transformed parameters
        )
    end
end

"""
    sample(s::BayesNegativeBinomial.Sampler; kwargs...)

Draw a posterior sample using the Gibbs sampler `s`, 
following Polson et al. (2013).    

# Arguments

* `mcmcsize = 2000`: the total number of MCMC iterations.
* `burnin = 1000`: the number of *burn-in* iterations.

# Example 

```julia
julia> using Random                 
julia> rng = MersenneTwister(1)     
julia> X = randn(100, 2)
julia> y = rand(0:2, 10)        
julia> sampler = BayesNegativeBinomial.Sampler(y, X)        
julia> chain = BayesNegativeBinomial.sample(sampler)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.    
"""
function sample(sampler::Sampler; mcmcsize = 10000, burnin = 5000)
    chain = [zeros(size(sampler.X, 2)) for _ in 1:(mcmcsize - burnin)]
    for iter in 1:mcmcsize
        step!(sampler)
        if iter > burnin
            chain[iter - burnin] .= sampler.β
        end
    end
    return chain
end

"""
    step!(s::BayesNegativeBinomial.Sampler)

Perform 1 iteration of the Gibbs sampler `s`, following Polson et al. (2013).

# Example 

```julia
julia> using Random, Distributions
julia> rng = MersenneTwister(1)
julia> X = randn(100, 2)
julia> y = rand(0:2, 10)
julia> s = BayesNegativeBinomial.Sampler(y, X)
julia> BayesNegativeBinomial.step!(s)
```

# References

1. Polson, N., Scott, J. & Windle, J. (2013) Bayesian inference for logistic 
    models using Pólya–Gamma latent variables, *Journal of the American 
    Statistical Association*, 108:504, 1339-1349,
    <https://doi.org/10.1080/01621459.2013.829001>.
"""
function step!(sampler::Sampler)
    step_ϕ!(sampler)
    step_s!(sampler)
    step_ω!(sampler)
    step_A!(sampler)
    step_b!(sampler)    
    step_g!(sampler)
    step_β!(sampler)
    return nothing
end

function step_ω!(sampler::Sampler)
    (; N, y, ω, ξ, s) = sampler
    step_ξ!(sampler)
    for i in 1:N
        ω[i] = rand(PolyaGammaPSWSampler(y[i] + s[], ξ[i]))
    end    
    return nothing
end

function step_g!(sampler::Sampler)
    sampler.update_g && return nothing
    (; D, gexp, update_g, mapping, g, μ0β, Σ0β, ζ0g) = sampler
    step_A!(sampler)
    step_b!(sampler)
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

function step_β!(sampler::Sampler)
    (; mapping, β, g) = sampler
    step_A!(sampler)
    step_b!(sampler)
    D = length(β)
    gexp = zeros(Bool, D)
    for d in 1:length(mapping)
        gexp[mapping[d]] .= g[d]
    end        
    β .= 0.0
    m1, Σ1 = posterior_hyperparameters(sampler)
    β[gexp] .= rand(MvNormal(m1, Σ1))
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

function step_ϕ!(sampler::Sampler)
    sampler.update_s && return nothing
    (; ξ, ϕ) = sampler
    @. ϕ = 1.0 / (1.0 + exp(ξ))
    return nothing
end

function step_s!(sampler::Sampler)
    sampler.update_s && return nothing
    (; N, q, ϕ, s, a0s, b0s) = sampler
    step_q!(sampler)
    a1s = a0s + sum(q)
    b1s = b0s
    for i in 1:N
        b1s - log(1 - ϕ[i]) 
    end
    ds = Gamma(a1s, 1 / b1s)
    s[] = rand(ds)
    return nothing
end

function step_ξ!(sampler::Sampler)
    (; ξ, X, β) = sampler
    mul!(ξ, X, β)
    return nothing
end

function step_q!(sampler::Sampler)
    (; N, y, q, s) = sampler
    s0 = s[]
    for i in 1:N
        q[i] = 0
        for j in 1:y[i]
            q[i] += rand(rng) <= s0 / (s0 + j - 1)
        end
    end
    return nothing
end

