module BayesNegativeBinomial

using Distributions
using PolyaGammaSamplers
using LinearAlgebra
using Random
using WomackDistribution

import Distributions: pdf, logpdf

include("body.jl")

end
