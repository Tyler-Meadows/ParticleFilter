var documenterSearchIndex = {"docs":
[{"location":"AnotherPage.html#The-ParticleFilter-Module","page":"Another page","title":"The ParticleFilter Module","text":"","category":"section"},{"location":"AnotherPage.html","page":"Another page","title":"Another page","text":"ParticleFilter","category":"page"},{"location":"AnotherPage.html#ParticleFilter","page":"Another page","title":"ParticleFilter","text":"ParticleFilter.jl\n\nThis is the bones for a discrete-time particle filter that I built to infer the infection numbers based on wastewater surveillance data\n\n\n\n\n\n","category":"module"},{"location":"AnotherPage.html#Module-Index","page":"Another page","title":"Module Index","text":"","category":"section"},{"location":"AnotherPage.html","page":"Another page","title":"Another page","text":"Modules = [ParticleFilter]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"AnotherPage.html#Detailed-API","page":"Another page","title":"Detailed API","text":"","category":"section"},{"location":"AnotherPage.html","page":"Another page","title":"Another page","text":"Modules = [ParticleFilter]\nOrder   = [:constant, :type, :function, :macro]","category":"page"},{"location":"AnotherPage.html#ParticleFilter.Filter","page":"Another page","title":"ParticleFilter.Filter","text":"Particle(T::Real,particles::Vector{Particle},Measurements::DataFrame,MeasurementModel::Function,DynamicModel::Function)\n\nT - Time\nparticles - Vector of particles\nMeasurements - DataFrame of measurements\nMeasurementModel - A function that takes a DataFrameRow and a particle and outputs a real number\nDynamicModel - An inplace function that mutates a particle\n\n\n\n\n\n","category":"type"},{"location":"AnotherPage.html#ParticleFilter.Particle","page":"Another page","title":"ParticleFilter.Particle","text":"Particle(weight::Float64,state::Vector{:<Real},pars::NamedTuple)\n\n\n\n\n\n","category":"type"},{"location":"AnotherPage.html#ParticleFilter.Particle-Tuple{Float64, Particle}","page":"Another page","title":"ParticleFilter.Particle","text":"Particle(w::Float64,p::Particle)\n\nConstruct a particle using a new weight and an old particle\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.average_particle-Tuple{Filter}","page":"Another page","title":"ParticleFilter.average_particle","text":"average_particle(p::Filter)\n\nDetermines the average particle from given particle filter\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.avg_pars-Tuple{Filter}","page":"Another page","title":"ParticleFilter.avg_pars","text":"avg_pars(p::Filter)\n\nFind the average of the parameters\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.max_weight-Tuple{Filter}","page":"Another page","title":"ParticleFilter.max_weight","text":"max_weight(p::Filter)\n\nReturn the particle with the highest weight\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.multinomial_sampling!-Tuple{Filter}","page":"Another page","title":"ParticleFilter.multinomial_sampling!","text":"multinomial_sampling!(p::Filter)\n\nResamples particles using a multinomial sampling procedure. If n is the number of particles, then this draws n samples from a multinomial distribution defined using the weights of the particles\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.normalize_weights!-Tuple{Filter}","page":"Another page","title":"ParticleFilter.normalize_weights!","text":"normalize_weights(p::Filter)\n\nNormalize the weights\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.percentile_sampling!-Tuple{Filter, Any}","page":"Another page","title":"ParticleFilter.percentile_sampling!","text":"percentile_sampling(particle_filter,measurement;q::Float64)\n\nresamples the particles based on the current measurement. Particles with a weight in the lowest q-th percentile are dropped and replaced by particles with the highest percentile.\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.propogate_sample!-Tuple{Filter, Particle}","page":"Another page","title":"ParticleFilter.propogate_sample!","text":"propogate_sample!(p::Filter,sample::Particle)\n\nUses the dynamic model to push particles forward one time step.\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.quant-Tuple{Filter, Any}","page":"Another page","title":"ParticleFilter.quant","text":"quant(particle_filter,q::Float64)\n\nDetermines the weight that separates the particles into two groups\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.systematic_resampling!-Tuple{Filter}","page":"Another page","title":"ParticleFilter.systematic_resampling!","text":"systematic_resampling!(p::Filter)\n\nApplies the systematic resampling procedure described in the POMP documentation by Aaron King et al. \n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.update!-Tuple{Filter}","page":"Another page","title":"ParticleFilter.update!","text":"update!(p::Filter)\n\nMove the particle filter forward one time-step. The particles are propogated using the dynamic model, and their weights updated using the measurement model. The particle  weights are normalized so that they sum to 1, and then particles are resampled from a multinomial distribution.\n\n\n\n\n\n","category":"method"},{"location":"AnotherPage.html#ParticleFilter.update_likelihood!-Tuple{Filter, Particle}","page":"Another page","title":"ParticleFilter.update_likelihood!","text":"update_likelihood!(p::filter,sample::particle)\n\nChanges the weight of the particle to its likelihood according to the measurement model\n\n\n\n\n\n","category":"method"},{"location":"index.html#ParticleFilter.jl","page":"Index","title":"ParticleFilter.jl","text":"","category":"section"},{"location":"index.html","page":"Index","title":"Index","text":"Documentation for ParticleFilter.jl","category":"page"}]
}
