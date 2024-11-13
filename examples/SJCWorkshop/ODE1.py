import pybamm
import numpy as np

# define variables
xn = pybamm.Variable("Negative electrode stochiometry")
xp = pybamm.Variable("Positive electrode stochiometry")
# define parameters
xn0 = pybamm.Parameter("Initial negative electrode stochiometry")
xp0 = pybamm.Parameter("Initial positive electrode stochiometry")
Qn = pybamm.Parameter("Negative electrode capacity [A.h]")
Qp = pybamm.Parameter("Positive electrode capacity [A.h]")
R = pybamm.Parameter("Resistance [Ohm]")
I = pybamm.FunctionParameter("Current", {"Time [s]": pybamm.t})
Un = pybamm.FunctionParameter("Negative OCV", {"Negative electrode stochiometry": xn})
Up = pybamm.FunctionParameter("Positive OCV", {"Positive electrode stochiometry": xp})
# define the model
model = pybamm.BaseModel("Reservoir model")
model.rhs = {xn:-I/Qn,xp:I/Qp}
model.initial_conditions = {xn:xn0,xp:xp0}
model.variables = {"Negative electrode stochiometry":xn,
                    "Positive electrode stochiometry":xp,
                      "Voltage": Up - Un - I*R}
# define the events
stop_at_xn_equal_0 = pybamm.Event("Stop at xn = 0", xn)
stop_at_xn_equal_1 = pybamm.Event("Stop at xn = 1", 1-xn)
stop_at_xp_equal_0 = pybamm.Event("Stop at xp = 0", xp)
stop_at_xp_equal_1 = pybamm.Event("Stop at xp = 1", 1-xp)
model.events = [stop_at_xn_equal_0,stop_at_xn_equal_1,stop_at_xp_equal_0,stop_at_xp_equal_1]
# define the parameter values
def graphite_LGM50_ocp_Chen2020(sto):
  u_eq = (
      1.9793 * np.exp(-39.3631 * sto)
      + 0.2482
      - 0.0909 * np.tanh(29.8538 * (sto - 0.1234))
      - 0.04478 * np.tanh(14.9159 * (sto - 0.2769))
      - 0.0205 * np.tanh(30.4444 * (sto - 0.6103))
  )

  return u_eq

def nmc_LGM50_ocp_Chen2020(sto):
  u_eq = (
      -0.8090 * sto
      + 4.4875
      - 0.0428 * np.tanh(18.5138 * (sto - 0.5542))
      - 17.7326 * np.tanh(15.7890 * (sto - 0.3117))
      + 17.5842 * np.tanh(15.9308 * (sto - 0.3120))
  )
  
  return u_eq

param = pybamm.ParameterValues({
    "Current": lambda t: 1 + 0.5 * pybamm.sin(100*t),
    "Initial negative electrode stochiometry": 0.9,
    "Initial positive electrode stochiometry": 0.1,
    "Negative electrode capacity [A.h]": 1,
    "Positive electrode capacity [A.h]": 1,
    "Resistance [Ohm]": 0.1,
    "Positive OCV": nmc_LGM50_ocp_Chen2020,
    "Negative OCV": graphite_LGM50_ocp_Chen2020,
})
# Solve the model
sim = pybamm.Simulation(model, parameter_values=param)
sol = sim.solve([0, 1])
sol.plot(["Voltage", "Negative electrode stochiometry", "Positive electrode stochiometry"])

