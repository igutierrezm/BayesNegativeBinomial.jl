using BayesNegativeBinomial
using Documenter

DocMeta.setdocmeta!(BayesNegativeBinomial, :DocTestSetup, :(using BayesNegativeBinomial); recursive=true)

makedocs(;
    modules=[BayesNegativeBinomial],
    authors="Iván Gutiérrez <ivangutierrez1988@gmail.com> and contributors",
    repo="https://github.com/igutierrezm/BayesNegativeBinomial.jl/blob/{commit}{path}#{line}",
    sitename="BayesNegativeBinomial.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://igutierrezm.github.io/BayesNegativeBinomial.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/igutierrezm/BayesNegativeBinomial.jl",
)
