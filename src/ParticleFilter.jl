"""
    ParticleFilter.jl

This is the bones for a discrete-time particle filter that I built to infer the infection numbers based on wastewater surveillance data

"""
module ParticleFilter
export Particle, Filter
export update!, average_particle
export FilterHistory, run_filter!
export log_likelihood, iterated_filtering!
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
    init_filter::Function
end

"""
    FilterHistory(T::Vector{Real},particles::Vector{Vector{Particle}})
"""
mutable struct FilterHistory
    T::Vector{Real}
    particles::Vector{Vector{Particle}}
end

function FilterHistory(p::Filter)
    FilterHistory([p.T],[p.particles])
end

"""
    reinitialize!(p::Filter)
    Restarts the particle filter in the case that the filter
    deviates too much from predictions. 

"""
function reinitialize!(p::Filter)
    N = length(p.particles)
    stats_pars = p.particles[1].stats_pars
    pars = avg_pars(p)
    p.init_filter(p,
                  N,
                  pars,
                  stats_pars)
end
"""
    average_particle(p::Filter)
Determines the average particle from given particle filter
"""
function average_particle(p::Filter)
    sum_weights = [p.weight for p in p.particles] |> sum
    avg_state = ones(size(p.particles[1].state))
    for i in eachindex(avg_state)
        avg_state[i] = [w.weight/sum_weights * w.state[i] for w in p.particles] |> sum
    end
    return Particle(1.0,avg_state,avg_pars(p),p.particles[1].stats_pars)
end

"""
    average_particle(p::Vector{Particle})
Find the average particle from a vector of particles
"""
function average_particle(p::Vector{Particle})
    sum_weights = [part.weight for part in p] |> sum
    avg_state = ones(size(p[1].state))
    for i in eachindex(avg_state)
        avg_state[i] = [w.weight/sum_weights *w.state[i] for w in p] |> sum
    end
    return Particle(sum_weights,avg_state,avg_pars(p),p[1].stats_pars)
end

"""
    avg_pars(p::Filter)
Find the average of the parameters
"""
function avg_pars(p::Filter)
    weights = [par.weight for par in p.particles]
    weights = weights./sum(weights)
    vals = sum(weights.*([values(p.pars)...] for p in p.particles))
    🔑 = keys(p.particles[1].pars)
    return (; zip(🔑,vals)...)
end

"""
    avg_pars(p::Vector{Particle})
Compute the average parameters from a vector of particles
"""
function avg_pars(p::Vector{Particle})
    weights = [par.weight for par in p]
    weights = weights./sum(weights)
    vals = sum(weights.*([values(par.pars)...] for par in p))
    🔑 = keys(p[1].pars)
    return (; zip(🔑,vals)...)
end

"""
    max_weight(p::Filter)
Return the particle with the highest weight
"""
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
Changes the weight of the particle to its likelihood according to the measurement model
"""
function update_likelihood!(p::Filter,sample::Particle)
    sample.weight = p.MeasurementModel(p.Measurements[p.T,:],sample)
end

 

"""
    quant(particle_filter,q::Float64)
Determines the weight that separates the particles into two groups
"""
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
        m === nothing && (m=length(weights))
        push!(p.particles, deep_copy(p.particles[m]))
    end
end


"""
    multinomial_sampling!(p::Filter)

Resamples particles using a multinomial sampling procedure.
If n is the number of particles, then this draws n samples from a multinomial distribution defined
using the weights of the particles
"""
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
        push!(new_samples, deepcopy(p.particles[m]))
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
        push!(new_samples,deepcopy(p.particles[m]))
    end
    p.particles .= new_samples
end




needs_resampling(p::Filter,threshold) = max_weight(p) < threshold
needs_resampling(p::Filter) = true

"""
    normalize_weights(p::Filter)
Normalize the weights
"""
function normalize_weights!(p::Filter)
    sum_weights = [w.weight for w in p.particles] |> sum
     if sum_weights < 1e-20
         reinitialize!(p)
     else    
        for w in p.particles
            w.weight /= sum_weights
        end
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
    p.T +=1
    @Threads.threads for part in p.particles
        propogate_sample!(p,part)
        update_likelihood!(p,part)
    end
    normalize_weights!(p)
    needs_resampling(p) && resample(p)
    nothing
end

"""
    update_history!(History::FilterHistory,p::Filter)
Adds the current state of the particle filter to the filter history.
"""
function update_history!(History::FilterHistory,p::Filter)
    push!(History.T,p.T)
    push!(History.particles,deepcopy(p.particles))
end

"""
    run_filter!(p::Filter)
updates the filter so that T is equal to the number of rows in the measurements dataframe. 
Returns a FilterHistory
"""
function run_filter!(p::Filter)
    History = FilterHistory(p)
    while p.T < nrow(p.Measurements)
        update!(p)
        update_history!(History,p)
    end
    return History
end



"""
    likelihood(H::FilterHistory)
Computes an estimate of the log likelihood of the filter.
"""
function log_likelihood(H::FilterHistory)
    ℒ = Float64[]
    for state in H.particles
        ℓ = mean([p.weight for p in state])
        push!(ℒ,log(ℓ))
    end
    return sum(ℒ)
end


import Base.getindex
function getindex(H::FilterHistory,N::UnitRange)
    return FilterHistory(H.T[N],H.particles[N])
end
function getindex(H::FilterHistory,N::Int)
    return FilterHistory([H.T[N]],[H.particles[N]])
end

"""
    varypars!(p::Filter,V::Vector{Distribution})

For the iterated filtering algorithm.
"""
function varypars!(p::Filter, V::Vector{Float64}; Dist = MvNormal)
    @assert length(p.particles[1].pars) == length(V)
    for part in p.particles
        setfield!(part,:pars, (;zip(keys(part.pars),Dist([values(part.pars)...],V)|>rand)...))
    end
end

"""
    iterated_filtering(p::Filter,)

Implements an iterated particle filter, which varies the parameters
along with the state variables to try to converge to parameters that
give the maximum log likelihood. 

Returns parameters
"""
function iterated_filtering!(p::Filter,
                            init_filter::Function,
                            N_particles::Int64,
                            M_iterations::Int64,
                            init_parameters,
                            statistical_parameters,
                            cooling_fraction::Float64,
                            randomwalk_σ::Vector{Float64},
                            ;Dist = MvNormal,
                            Track_Likelihood = false)
        Track_Likelihood && (ℒ = [])
        for m in 1:M_iterations
            p.T = 1
            if m == 1
                init_filter(p,N_particles,init_parameters,statistical_parameters)
            else
                init_filter(p)
            end
            varypars!(p,randomwalk_σ;Dist=Dist)
            if Track_Likelihood
                History = FilterHistory(p)
            end
            while p.T < nrow(p.Measurements)
                update!(p)
                varypars!(p,randomwalk_σ;Dist=Dist)
                Track_Likelihood && update_history!(History,p)
            end
            Track_Likelihood && push!(ℒ,log_likelihood(History))
            normalize_weights!(p)
            randomwalk_σ = randomwalk_σ*cooling_fraction^(2*m/50)
            print("Iteration $m of $M_iterations \r")
        end
    if Track_Likelihood
        return average_particle(p).pars, ℒ
    else    
        return average_particle(p).pars
    end 
end


end # module ParticleFilter