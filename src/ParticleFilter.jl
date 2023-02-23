"""
    ParticleFilter.jl

This is the bones for a discrete-time particle filter that I built to infer the infection numbers based on wastewater surveillance data

"""
module ParticleFilter
export Particle, Filter
export update!, average_particle

# Required packages
using Distributions
using DataFrames

#= Structures and Objects =#

"""
    Particle(weight::Float64,state::Vector{:<Real},pars::NamedTuple)

"""
mutable struct Particle
    weight::Float64
    state::Vector{<:Real}
    pars::NamedTuple
    stats_pars::NamedTuple
end

"""
    Particle(w::Float64,p::Particle)
Construct a particle using a new weight and an old particle
"""
function Particle(w::Float64,p::Particle)
    Particle(w,p.state,p.pars,p.stats_pars)
end

function Particle(p::Particle)
    Particle(p.weight,p.state,p.pars,p.stats_pars)
end

""" 
    Filter(T::Real,particles::Vector{Particle},Measurements::DataFrame,MeasurementModel::Function,DynamicModel::Function)
 * T - Time
 * particles - Vector of particles
 * Measurements - DataFrame of measurements
 * MeasurementModel - A function that takes a DataFrameRow and a particle and outputs a real number
 * DynamicModel - An inplace function that mutates a particle
"""
mutable struct Filter
    T::Real
    particles::Vector{Particle}
    Measurements::DataFrame
    MeasurementModel::Function
    DynamicModel::Function
end

"""
    FilterHistory(T::Vector{Real},particles::Vector{Vector{Particle}})
"""
mutable struct FilterHistory
    T::Vector{Real}
    particles::Vector{Vector{Real}}
end

function FilterHistory(p::Filter)
    FilterHistory([p.T],[p.particles])
end

"""
    average_particle(p::Filter)
Determines the average particle from given particle filter"""
function average_particle(p::Filter)
    sum_weights = [p.weight for p in p.particles] |> sum
    avg_state = ones(size(p.particles[1].state))
    for i in 1:length(avg_state)
        avg_state[i] = [w.weight/sum_weights * w.state[i] for w in p.particles] |> sum
    end
    return Particle(1.0,avg_state,avg_pars(p),p.particles[1].stats_pars)
end

"""
    avg_pars(p::Filter)
Find the average of the parameters
"""
function avg_pars(p::Filter)
    weights = [par.weight for par in p.particles]
    vals = sum(weights.*([values(p.pars)...] for p in p.particles))
    🔑 = keys(p.particles[1].pars)
    return (; zip(🔑,vals)...)
end

"""
    max_weight(p::Filter)
Return the particle with the highest weight"""
max_weight(p::Filter) = Base.maximum([w.weight for w in p.particles])

"""
    propogate_sample!(p::Filter,sample::Particle)
Uses the dynamic model to push particles forward one time step.
"""
function propogate_sample!(p::Filter,sample::Particle)
    du = zero(sample.state)
    p.DynamicModel(du,sample.state,p.T,sample.pars)
    sample.state += du
end

"""
    update_likelihood!(p::filter,sample::particle)
Changes the weight of the particle to its likelihood according to the measurement model"""
function update_likelihood!(p::Filter,sample::Particle)
    sample.weight = p.MeasurementModel(p.Measurements[p.T,:],sample)
end

 

"""
    quant(particle_filter,q::Float64)
Determines the weight that separates the particles into two groups"""
function quant(p::Filter,q)
    tmp = [w.weight for w in p.particles]
    quantile(tmp,q)
end

"""
    percentile_sampling(particle_filter,measurement;q::Float64)

resamples the particles based on the current measurement.
Particles with a weight in the lowest q-th percentile are dropped
and replaced by particles with the highest percentile.
"""
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


"""
    multinomial_sampling!(p::Filter)

Resamples particles using a multinomial sampling procedure.
If n is the number of particles, then this draws n samples from a multinomial distribution defined
using the weights of the particles"""
function multinomial_sampling!(p::Filter)
    N_particles = length(p.particles)
    weights = [w.weight for w in p.particles]
    weights = weights./(sum(weights))
    Q = cumsum(weights)
    new_samples = []
    for n in 1:N_particles
        u = Uniform(0.0,1.0) |> rand
        m = findfirst(x->x>u,Q)
        isnothing(m) && (m=N_particles)
        push!(new_samples, Particle(p.particles[m]))
    end
    p.particles .= new_samples
end

""" systematic_resampling!(p::Filter)

Applies the systematic resampling procedure described in the
POMP documentation by Aaron King et al. 
"""
function systematic_resampling!(p::Filter)
    N_particles = length(p.particles)
    weights = [w.weight for w in p.particles]
    weights = weights./(sum(weights))
    Q = cumsum(weights)
    new_samples = []
    U1 = Uniform(0,1/N_particles) |> rand
    U = U1.+0:(1/N_particles):1
    for Uj in U
        m = findfirst(Uj.<Q)
        isnothing(m) && (m =N_particles) 
        push!(new_samples,Particle(p.particles[m]))
    end
    p.particles .= new_samples
end




needs_resampling(p::Filter,threshold) = max_weight(p) < threshold
needs_resampling(p::Filter) = true

"""
    normalize_weights(p::Filter)
Normalize the weights"""
function normalize_weights!(p::Filter)
    sum_weights = [w.weight for w in p.particles] |> sum
        for w in p.particles
            w.weight /= sum_weights
        end
end

""" 
    update!(p::Filter)

Move the particle filter forward one time-step. The particles are propogated using
the dynamic model, and their weights updated using the measurement model. The particle 
weights are normalized so that they sum to 1, and then particles are resampled from a multinomial
distribution.
"""
function update!(p::Filter;resample=systematic_resampling!)
    @Threads.threads for part in p.particles
        propogate_sample!(p,part)
        update_likelihood!(p,part)
    end
    normalize_weights!(p)
    needs_resampling(p) && resample(p)
    p.T +=1
    nothing
end

function update_history!(History::FilterHistory,p::Filter)
    push!(History.T,[p.T])
    push!(History.particles,[p.particles])
end


end # module ParticleFilter
