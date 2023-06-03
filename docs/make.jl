using NeuralNetworks
using Documenter

DocMeta.setdocmeta!(NeuralNetworks, :DocTestSetup, :(using NeuralNetworks); recursive=true)

makedocs(;
    modules=[NeuralNetworks],
    authors="singularitti <singularitti@outlook.com> and contributors",
    repo="https://github.com/singularitti/NeuralNetworks.jl/blob/{commit}{path}#{line}",
    sitename="NeuralNetworks.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://singularitti.github.io/NeuralNetworks.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/singularitti/NeuralNetworks.jl",
    devbranch="main",
)
