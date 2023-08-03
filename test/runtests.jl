using Test, ParticleFilter

p = Particle(0.1,[1.0],(σ = 1, ),(μ = 0.0, σ = 0.01,))

@test p.pars.σ == 1
