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
        mathengine=Documenter.HTMLWriter.MathJax3(),
    ),
    pages=[
        "Home" => "index.md",
        "Manual" => [
            "Installation Guide" => "man/installation.md",
            "Backpropagation" => "man/backpropagation.md",
            "Troubleshooting" => "man/troubleshooting.md",
        ],
        "Reference" => Any[
            "Public API" => "lib/public.md",
            # "Internals" => map(
            #     s -> "lib/internals/$(s)",
            #     sort(readdir(joinpath(@__DIR__, "src/lib/internals")))
            # ),
        ],
        "Developer Docs" => [
            "Contributing" => "developers/contributing.md",
            "Style Guide" => "developers/style-guide.md",
            "Design Principles" => "developers/design-principles.md",
        ],
    ],
)

deploydocs(;
    repo="github.com/singularitti/NeuralNetworks.jl",
    devbranch="main",
)
