module ParticleFilter

# Required packages
using Distributions

export Particle, Filter
export update!, average_particle

##= Structures and Objects =##
# Particle struct, holds weight, state and parameters.
mutable struct Particle
    weight::Float64
    state::Vector{<:Real}
    pars::NamedTuple
end

# particle constructor
function Particle(w,p::Particle)
    Particle(w,p.state,p.pars)
end

# ParticleFilter Struct
mutable struct Filter
    N::Int64
    particles::Vector{Particle}
    dist::Distribution
    pars::NamedTuple
end

function average_particle(p::Filter)
    sum_weights = [p.weight for p in p.particles] |> sum
    avg_state = ones(size(p.particles[1].state))
    for i in 1:length(avg_state)
        avg_state[i] = [w.weight/sum_weights * w.state[i] for w in p.particles] |> sum
    end
    return Particle(1.0,avg_state,p.pars)
end

max_weight(p::Filter) = Base.maximum([w.weight for w in p.particles])

function propogate_sample!(p::Filter,sample,fn::Function)
    du = zero(sample.state)
    fn(du,sample.state,~,p.pars)
    sample.state += du
end

function propogate_sample!(p::Filter,sample::Particle,fn::Function)
    du = zero(sample.state)
    fn(du,sample.state,~,sample.pars)
    sample.state += du
end

function propogate_sample!(p::Filter,sample::Particle,fn::Function,t::Int64)
    du = zero(sample.state)
    fn(du,sample.state,t,sample.pars)
    sample.state += du
end

function compute_likelihood(p::Filter,sample,measurement)
    likelihood(p.dist,measurement,sample.state[2])
end

function update_likelihood!(p::Filter,sample::Particle,measurement)
    sample.weight = likelihood(p.dist,measurement,sample.state[2])
end


## Remove the lowest weighted 10% of particles
function quant(p::Filter,q)
    tmp = [w.weight for w in p.particles]
    quantile(tmp,q)
end

#Remove particles with 0 weight, and those too far from the measurement
function percentile_sampling!(p::Filter,Virus;q=0.1)
    filter!(x -> x.weight ≠ 0.0, p.particles)
    filter!(x -> x.weight ≥ quant(p,q),p.particles)
    normalize_weights!(p,Virus)
    weights = [part.weight for part in p.particles]
    Q = cumsum(weights)
    for j in (length(p.particles)+1):p.N
        m = findfirst(x->x<rand(),Q)
        m == nothing && (m=length(weights))
        push!(p.particles, p.particles[m])
    end
end

function multinomial_sampling!(p::Filter)
    weights = [w.weight for w in p.particles]
    Q = cumsum(weights)
    new_samples = []
    for n in 1:p.N
        u = Uniform(0.0,1.0) |> rand
        m = findfirst(x->x>u,Q)
        isnothing(m) && (m=length(weights))
        push!(new_samples, Particle(1.0/p.N,p.particles[m]))
    end
    p.particles .= new_samples
end

needs_resampling(p::Filter,threshold) = max_weight(p) > threshold
needs_resampling(p::Filter) = true

function normalize_weights!(p::Filter,Virus;verbose=false)
    sum_weights = [w.weight for w in p.particles] |> sum
    if sum_weights < 1e-10
        verbose && print("sum of weights too low \n")
        init_particles!(p,Virus)
    else
        for w in p.particles
            w.weight /= sum_weights
        end
    end
end


function update!(p::Filter,
    measurement,
    fn::Function)
    Threads.@threads for part in p.particles
        propogate_sample!(p,part,fn)
        update_likelihood!(p,part,measurement)
    end
    nothing
end
function update!(p::Filter,
    measurement,
    fn::Function,t)
    Threads.@threads for part in p.particles
        propogate_sample!(p,part,fn,t)
        update_likelihood!(p,part,measurement)
    end
    nothing
end

function update!(p::Filter,
        measurement::Missing,
        fn::Function)
    Threads.@threads for part in p.particles
        propogate_sample!(p,part,fn)
    end
    nothing
end

function update!(p::Filter,
    measurement::Missing,
    fn::Function,t)
    Threads.@threads for part in p.particles
        propogate_sample!(p,part,fn,t)
    end
    nothing
end


end # module ParticleFilter
