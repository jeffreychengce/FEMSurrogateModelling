import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad, jit, vmap, lax
import jax.scipy as jsp
import jax.scipy.optimize as jsp_opt
import optax 
import jaxopt
from jaxopt import ScipyBoundedMinimize

import jax

@jit
def mpm(E):
    # nsteps
    nsteps = 10
    
    # mom tolerance
    # tol = 1e-12

    # Domain length
    Lx = 25
    Ly = 25

    # Material properties
    rho = 1

    # Computational grid
    nelementsx = 13
    nelementsy = 13
    nelements = nelementsx * nelementsy
    
    dx = Lx / nelementsx
    dy = Lx / nelementsy
    
    # Create equally spaced nodes
    x_n, y_n = jnp.meshgrid(jnp.linspace(0, Lx, nelementsx + 1), jnp.linspace(0, Ly, nelementsy + 1))
    x_n = x_n.flatten()
    y_n = y_n.flatten()
    nnodes = len(x_n)
    
    
    # Set-up a 2D array of elements with node ids
    elements = jnp.zeros((nelements, 4), dtype=int)
    for nidx in range(nelementsx):
        for nidy in range(nelementsy):
            nid = nidx * nelements + nidy
            elements = elements.at[nid, 0].set(nidx * nelements + nidy)
            elements = elements.at[nid, 1].set(nidx * nelements + nidy + 1)
            elements = elements.at[nid, 2].set((nidx + 1) * nelements + nidy)
            elements = elements.at[nid, 3].set((nidx + 1) * nelements + nidy + 1)
            
    # Loading conditions
    v0x = 0.1              # initial velocity
    v0y = 0.1              # initial velocity
    c  = jnp.sqrt(E/rho)   # speed of sound
    b1 = jnp.pi / (2 * Lx) # beta1
    b2 = jnp.pi / (2 * Ly) # beta2
    # w1 = b1 * c            # omega1
    # w2 = b2 * c            # omega2
    
    # Create material points at the center of each element
    nparticles = nelements  # number of particles
    
    # Id of the particle in the central element
    pmid = nelements // 2  # Midpoint of the material points
    
    # Material point properties
    x_p       = jnp.zeros(nparticles)            # positions
    y_p       = jnp.zeros(nparticles)            # positions
    vol_p     = jnp.ones(nparticles) * dx * dy   # volume
    mass_p    = vol_p * rho                      # mass
    stress_px = jnp.zeros(nparticles)            # stress
    stress_py = jnp.zeros(nparticles)            # stress
    vel_px    = jnp.zeros(nparticles)            # velocity
    vel_py    = jnp.zeros(nparticles)            # velocity
    
    # Create particle at the center
    x_p      = 0.5 * (x_n[:-1] + x_n[1:])
    y_p      = 0.5 * (y_n[:-1] + y_n[1:])
    # set initial velocities
    vel_px   = v0x * jnp.sin(b1 * x_p)
    vel_py   = v0y * jnp.sin(b2 * y_p)
    
    # Time steps and duration
    # dt_crit = jnp.max(jnp.array([dx / c, dy / c]))
    dt = 0.02
    
    # results
    # tt = jnp.zeros(nsteps)
    vt = jnp.zeros((nsteps, 2))
    xt = jnp.zeros((nsteps, 2))
    
    def step(i, carry):
        x_p, y_p, mass_p, vel_px, vel_py, vol_p, stress_px, stress_py, vt, xt = carry
        # reset nodal values
        mass_n  = jnp.zeros(nnodes)   # mass
        mom_nx   = jnp.zeros(nnodes)  # momentum
        mom_ny   = jnp.zeros(nnodes)  # momentum
        fint_nx  = jnp.zeros(nnodes)  # internal force
        fint_ny  = jnp.zeros(nnodes)  # internal force

        # iterate through each element
        for eid in range(nelements):
            # get nodal ids
            nid1, nid2, nid3, nid4 = elements[eid]

            # compute shape functions and derivatives
            N1x = 1 - abs(x_p[eid] - x_n[nid1]) / dx
            N2x = 1 - abs(x_p[eid] - x_n[nid2]) / dx
            N3x = 1 - abs(x_p[eid] - x_n[nid3]) / dx
            N4x = 1 - abs(x_p[eid] - x_n[nid4]) / dx
            N1y = 1 - abs(y_p[eid] - y_n[nid1]) / dy
            N2y = 1 - abs(y_p[eid] - y_n[nid2]) / dy
            N3y = 1 - abs(y_p[eid] - y_n[nid3]) / dy
            N4y = 1 - abs(y_p[eid] - y_n[nid4]) / dy
            
            dN1x = -1/dx
            dN2x = 1/dx
            dN3x = -1/dx
            dN4x = 1/dx
            dN1y = -1/dy
            dN2y = 1/dy
            dN3y = -1/dy
            dN4y = 1/dy

            # map particle mass and momentum to nodes
            mass_n = mass_n.at[nid1].set(mass_n[nid1] + N1x * mass_p[eid])
            mass_n = mass_n.at[nid2].set(mass_n[nid2] + N2x * mass_p[eid])
            mass_n = mass_n.at[nid3].set(mass_n[nid3] + N3x * mass_p[eid])
            mass_n = mass_n.at[nid4].set(mass_n[nid4] + N4x * mass_p[eid])

            mom_nx = mom_nx.at[nid1].set(mom_nx[nid1] + N1x * mass_p[eid] * vel_px[eid])
            mom_nx = mom_nx.at[nid2].set(mom_nx[nid2] + N2x * mass_p[eid] * vel_px[eid])
            mom_nx = mom_nx.at[nid3].set(mom_nx[nid3] + N3x * mass_p[eid] * vel_px[eid])
            mom_nx = mom_nx.at[nid4].set(mom_nx[nid4] + N4x * mass_p[eid] * vel_px[eid])
            mom_ny = mom_ny.at[nid1].set(mom_ny[nid1] + N1y * mass_p[eid] * vel_py[eid])
            mom_ny = mom_ny.at[nid2].set(mom_ny[nid2] + N2y * mass_p[eid] * vel_py[eid])
            mom_ny = mom_ny.at[nid3].set(mom_ny[nid3] + N3y * mass_p[eid] * vel_py[eid])
            mom_ny = mom_ny.at[nid4].set(mom_ny[nid4] + N4y * mass_p[eid] * vel_py[eid])

            # compute nodal internal force
            fint_nx = fint_nx.at[nid1].set(fint_nx[nid1] - vol_p[eid] * stress_px[eid] * dN1x)
            fint_nx = fint_nx.at[nid2].set(fint_nx[nid2] - vol_p[eid] * stress_px[eid] * dN2x)
            fint_nx = fint_nx.at[nid3].set(fint_nx[nid3] - vol_p[eid] * stress_px[eid] * dN3x)
            fint_nx = fint_nx.at[nid4].set(fint_nx[nid4] - vol_p[eid] * stress_px[eid] * dN4x)
            fint_ny = fint_ny.at[nid1].set(fint_ny[nid1] - vol_p[eid] * stress_py[eid] * dN1y)
            fint_ny = fint_ny.at[nid2].set(fint_ny[nid2] - vol_p[eid] * stress_py[eid] * dN2y)
            fint_ny = fint_ny.at[nid3].set(fint_ny[nid3] - vol_p[eid] * stress_py[eid] * dN3y)
            fint_ny = fint_ny.at[nid4].set(fint_ny[nid4] - vol_p[eid] * stress_py[eid] * dN4y)
        
        # apply boundary conditions
        mom_nx = mom_nx.at[::nelementsx+1].set(jnp.zeros(nelementsy+1))  # Nodal velocity v = 0 in m * v at node 0.
        mom_ny = mom_ny.at[::nelementsy+1].set(jnp.zeros(nelementsx+1))  # Nodal velocity v = 0 in m * v at node 0.
        fint_nx = fint_nx.at[::nelementsx+1].set(jnp.zeros(nelementsy+1))  # Nodal force f = m * a, where a = 0 at node 0.
        fint_ny = fint_ny.at[::nelementsy+1].set(jnp.zeros(nelementsx+1))  # Nodal force f = m * a, where a = 0 at node 0.

        # update nodal momentum
        mom_nx = mom_nx + fint_nx * dt
        mom_ny = mom_ny + fint_ny * dt

        # update particle velocity position and stress
        # iterate through each element
        for eid in range(nelements):
            # get nodal ids
            nid1, nid2, nid3, nid4 = elements[eid]

            # compute shape functions and derivatives
            N1x = 1 - abs(x_p[eid] - x_n[nid1]) / dx
            N2x = 1 - abs(x_p[eid] - x_n[nid2]) / dx
            N3x = 1 - abs(x_p[eid] - x_n[nid3]) / dx
            N4x = 1 - abs(x_p[eid] - x_n[nid4]) / dx
            N1y = 1 - abs(y_p[eid] - y_n[nid1]) / dy
            N2y = 1 - abs(y_p[eid] - y_n[nid2]) / dy
            N3y = 1 - abs(y_p[eid] - y_n[nid3]) / dy
            N4y = 1 - abs(y_p[eid] - y_n[nid4]) / dy
            
            dN1x = -1/dx
            dN2x = 1/dx
            dN3x = -1/dx
            dN4x = 1/dx
            dN1y = -1/dy
            dN2y = 1/dy
            dN3y = -1/dy
            dN4y = 1/dy
            
            # compute particle velocity
            vel_px = vel_px.at[eid].set(vel_px[eid] + dt * N1x * fint_nx[nid1] / mass_n[nid1])
            vel_px = vel_px.at[eid].set(vel_px[eid] + dt * N2x * fint_nx[nid2] / mass_n[nid2])
            vel_px = vel_px.at[eid].set(vel_px[eid] + dt * N3x * fint_nx[nid3] / mass_n[nid3])
            vel_px = vel_px.at[eid].set(vel_px[eid] + dt * N4x * fint_nx[nid4] / mass_n[nid4])
            vel_py = vel_py.at[eid].set(vel_py[eid] + dt * N1y * fint_ny[nid1] / mass_n[nid1])
            vel_py = vel_py.at[eid].set(vel_py[eid] + dt * N2y * fint_ny[nid2] / mass_n[nid2])
            vel_py = vel_py.at[eid].set(vel_py[eid] + dt * N3y * fint_ny[nid3] / mass_n[nid3])
            vel_py = vel_py.at[eid].set(vel_py[eid] + dt * N4y * fint_ny[nid4] / mass_n[nid4])
            
            
            
            # update particle position based on nodal momentum
            x_p = x_p.at[eid].set(x_p[eid] + dt * (N1x * mom_nx[nid1]/mass_n[nid1] + N2x * mom_nx[nid2]/mass_n[nid2] + N3x * mom_nx[nid3]/mass_n[nid3] + N4x * mom_nx[nid4]/mass_n[nid4]))
            y_p = y_p.at[eid].set(y_p[eid] + dt * (N1y * mom_ny[nid1]/mass_n[nid1] + N2y * mom_ny[nid2]/mass_n[nid2] + N3y * mom_ny[nid3]/mass_n[nid3] + N4y * mom_ny[nid4]/mass_n[nid4]))

            # nodal velocity
            nv1x = mom_nx[nid1]/mass_n[nid1]
            nv2x = mom_nx[nid2]/mass_n[nid2]
            nv3x = mom_nx[nid3]/mass_n[nid3]
            nv4x = mom_nx[nid4]/mass_n[nid4]
            nv1y = mom_ny[nid1]/mass_n[nid1]
            nv2y = mom_ny[nid2]/mass_n[nid2]
            nv3y = mom_ny[nid3]/mass_n[nid3]
            nv4y = mom_ny[nid4]/mass_n[nid4]

             # rate of strain increment
            grad_vx = dN1x * nv1x + dN2x * nv2x + dN3x * nv3x + dN4x * nv4x
            grad_vy = dN1y * nv1y + dN2y * nv2y + dN3y * nv3y + dN4y * nv4y
            # particle dstrain
            dstrainx = grad_vx * dt
            dstrainy = grad_vy * dt
            # particle volume
            vol_p = vol_p.at[eid].set((1 + dstrainx) * (1 + dstrainy) * vol_p[eid])
            # update stress using linear elastic model
            stress_px = stress_px.at[eid].set(stress_px[eid] + E * dstrainx)
            stress_py = stress_py.at[eid].set(stress_py[eid] + E * dstrainy)

        # results
        vt = vt.at[i, 0].set(vel_px[pmid])
        vt = vt.at[i, 1].set(vel_py[pmid])
        xt = xt.at[i, 0].set(x_p[pmid])
        xt = xt.at[i, 1].set(y_p[pmid])

        return (x_p, y_p, mass_p, vel_px, vel_py, vol_p, stress_px, stress_py, vt, xt)

    x_p, y_p, mass_p, vel_px, vel_py, vol_p, stress_px, stress_py, vt, xt = lax.fori_loop(0, nsteps, step, (x_p, y_p, mass_p, vel_px, vel_py, vol_p, stress_px, stress_py, vt, xt))


    
    return vt

from jax.lib import xla_bridge
print(xla_bridge.get_backend().platform)

print('Calculating target trajectory')
Etarget = 100
target = mpm(Etarget)


#############################################################
#  NOTE: Uncomment the line only for TFP optimizer and 
#        jaxopt value_and_grad = True
#############################################################
# @jax.value_and_grad
@jit
def compute_loss(E):
    vt = mpm(E)
    return jnp.linalg.norm(vt - target)

# BFGS Optimizer
# TODO: Implement box constrained optimizer
def jaxopt_bfgs(params, niter):
  opt= jaxopt.BFGS(fun=compute_loss, value_and_grad=True, tol=1e-5, implicit_diff=False, maxiter=niter)
  res = opt.run(init_params=params)
  result, _ = res
  return result

# Optimizers
def optax_adam(params, niter):
  # Initialize parameters of the model + optimizer.
  start_learning_rate = 1e-1
  optimizer = optax.adam(start_learning_rate)
  opt_state = optimizer.init(params)

  # A simple update loop.
  for i in range(niter):
    print('iteration: ', i)
    grads = grad(compute_loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    print(i, params)
  return params
  
# Tensor Flow Probability Optimization library
def tfp_lbfgs(params):
  results = tfp.optimizer.lbfgs_minimize(
        jax.jit(compute_loss), initial_position=params, tolerance=1e-5)
  return results.position

# Initial model - Young's modulus 
params = 95.0

print('Running optimizer')
# vt = tfp_lbfgs(params)               # LBFGS optimizer
result = optax_adam(params, 10)     # ADAM optimizer

"""
f = jax.jit(compute_loss)
df = jax.jit(jax.grad(compute_loss))
E = 95.0
print(0, E)
for i in range(10):
    E = E - f(E)/df(E)
    print(i, E)
"""
print("E: {}".format(result))
vel = mpm(result)
# update time steps
dt = 0.02
nsteps = 10
tt = jnp.arange(0, nsteps) * dt


vel = vel.reshape((2,10))
target = target.reshape((2,10))


# Plot results
plt.plot(tt, vel[0,:], 'r', markersize=1, label='mpm')
plt.plot(tt, target[0,:], 'ob', markersize=1, label='mpm-target')
plt.xlabel('time (s)')
plt.ylabel('x velocity (m/s)')
plt.legend()
plt.show()

plt.plot(tt, vel[1,:], 'r', markersize=1, label='mpm')
plt.plot(tt, target[1,:], 'ob', markersize=1, label='mpm-target')
plt.xlabel('time (s)')
plt.ylabel('y velocity (m/s)')
plt.legend()
plt.show()

print(result)