import pybamm
import numpy as np
import matplotlib.pylab as plt

# define the parameters
D = pybamm.Parameter("Diffusion coefficient")
R = pybamm.Parameter("Particle radius")
j = pybamm.Parameter("Current")
F = pybamm.Scalar(96485)
c0 = pybamm.Parameter("Initial concentration")



#setting up the model
model = pybamm.BaseModel()

c = pybamm.Variable("Concentration", domain="negative particle")
N = -pybamm.grad(c)  # define the flux
dcdt = -pybamm.div(D*N)  # define the rhs equation

model.rhs = {c: dcdt}  # add the equation to rhs dictionary

# initial conditions
model.initial_conditions = {c: c0}

# boundary conditions
lbc = pybamm.Scalar(0)
rbc = -j/(F*D)
model.boundary_conditions = {c: {"left": (lbc, "Neumann"), "right": (rbc, "Neumann")}}

param = pybamm.ParameterValues({"Diffusion coefficient":3.9e-14,
                                "Particle radius":    1.0e-5,
                                "Current": 1.4,
                                "Initial concentration": 2.5e4
})

# variables we are interested in
model.variables = {"Concentration": c,
                   "Surface concentration": pybamm.surf(c),
                     "Flux": N}

# define geometry
r = pybamm.SpatialVariable(
    "r", domain=["negative particle"], coord_sys="spherical polar"
)
geometry = {"negative particle": {r: {"min": pybamm.Scalar(0), "max": R}}}

param.process_model(model)
param.process_geometry(geometry)

# mesh and discretise
submesh_types = {"negative particle": pybamm.Uniform1DSubMesh}
var_pts = {r: 20}
mesh = pybamm.Mesh(geometry, submesh_types, var_pts)

#discretize
spatial_methods = {"negative particle": pybamm.FiniteVolume()}
disc = pybamm.Discretisation(mesh, spatial_methods)
disc.process_model(model);

# solve
solver = pybamm.ScipySolver()
t = np.linspace(0, 3600, 60)
solution = solver.solve(model, t)


# post-process, so that the solution can be called at any time t or space r
# (using interpolation)
c = solution["Concentration"]
c_surf = solution["Surface concentration"]

# plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

ax1.plot(solution.t, c_surf(solution.t))
ax1.set_xlabel("Time [s]")
ax1.set_ylabel("Surface concentration [mol.m-3]")

r = mesh["negative particle"].nodes # radial position
time = 1000  # time in seconds
ax2.plot(r * 1e6, c(t=time, r=r), label="t={}[s]".format(time))
ax2.set_xlabel("Particle radius [microns]")
ax2.set_ylabel("Concentration [mol.m-3]")
ax2.legend()

plt.tight_layout()
plt.show()

