using Test, ParticleFilter

p = Particle(0.1,[1.0],(σ = 1, ))

@test p.pars.σ == 1
