using Documenter
using ParticleFilter

push!(LOAD_PATH,"../src/")
makedocs(
    sitename = "ParticleFilter.jl Documentation",
    pages = ["Index" => "index.md",],
    format = Documenter.HTML(prettyurls=false),
    modules = [ParticleFilter]
)

# Documenter can also automatically deploy documentation to gh-pages.
# See "Hosting Documentation" and deploydocs() in the Documenter manual
# for more information.
=deploydocs(
    repo = "https://github.com/Tyler-Meadows/ParticleFilter.jl.git",
    devbranch = "main"
)=#
