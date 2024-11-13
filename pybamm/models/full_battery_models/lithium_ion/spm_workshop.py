#
# Single Particle Model (SPM)
#
import pybamm


class WorkshopSPM(pybamm.BaseModel):
    """Single Particle Model (SPM) model of a lithium-ion battery, from
    :footcite:t:`Marquis2019`.

    Parameters
    ----------
    name : str, optional
        The name of the model.
    """

    def __init__(self, name="Single Particle Model"):
        super().__init__(name=name)

        # This command registers the article where the model comes from so it appears
        # when calling `pybamm.print_citations()`
        pybamm.citations.register("Marquis2019")

        ######################
        # Variables
        ######################
        # Define the variables of the model.
        # define state variables
        electrodes = ["negative", "positive"]
        c_i = [pybamm.Variable(f"{e.capitalize()} particle concentration [mol.m-3]", domain=f"{e} particle") for e in electrodes]


        ######################
        # Parameters
        ######################
        # You need to fill this section with the parameters of the model. To ensure
        # it works within PyBaMM, you need to use the same names as in PyBaMM's parameter
        # sets (e.g. see https://github.com/pybamm-team/PyBaMM/blob/develop/pybamm/input/parameters/lithium_ion/Chen2020.py)

        I = pybamm.FunctionParameter("Current function [A]", {"Time [s]": pybamm.t})
        D_i = [pybamm.Parameter(f"{e.capitalize()} particle diffusivity [m2.s-1]") for e in electrodes]
        k_i = [pybamm.Parameter(f"{e.capitalize()} electrode reaction rate [m.s-1]") for e in electrodes]
        R_i = [pybamm.Parameter(f"{e.capitalize()} particle radius [m]") for e in electrodes]
        c0_i = [pybamm.Parameter(f"Initial concentration in {e} electrode [mol.m-3]") for e in electrodes]
        c_i_max = [pybamm.Parameter(f"Maximum concentration in {e} electrode [mol.m-3]") for e in electrodes]
        delta_i = [pybamm.Parameter(f"{e.capitalize()} electrode thickness [m]") for e in electrodes]
        A = pybamm.Parameter("Electrode width [m]") * pybamm.Parameter("Electrode height [m]")  # PyBaMM takes the width and height of the electrodes (assumed rectangular) rather than the total area
        epsilon_i = [pybamm.Parameter(f"{e.capitalize()} electrode active material volume fraction") for e in electrodes]
        c_e = pybamm.Parameter("Initial concentration in electrolyte [mol.m-3]") 
        T = pybamm.Parameter("Ambient temperature [K]")

        # define universal constants (PyBaAMM has them built in)
        F = pybamm.constants.F
        R = pybamm.constants.R

        # define variables that depend on the parameters
        a_i = [3 * epsilon_i[i] / R_i[i] for i in [0, 1]]
        j_i = [I / a_i[0] / delta_i[0] / F / A, -I / a_i[1] / delta_i[1] / F / A]

        ######################
        # Particle model
        ######################
        # Define the governing equations for each particle.
        # governing equations
        dcdt_i = [pybamm.div(D_i[i] * pybamm.grad(c_i[i])) for i in [0, 1]]
        self.rhs = {c_i[i]: dcdt_i[i] for i in [0, 1]}

        # boundary conditions
        lbc = pybamm.Scalar(0)
        rbc = [-j_i[i] / D_i[i] for i in [0, 1]]
        self.boundary_conditions = {c_i[i]: {"left": (lbc, "Neumann"), "right": (rbc[i], "Neumann")} for i in [0, 1]}

        # initial conditions
        self.initial_conditions = {c_i[i]: c0_i[i] for i in [0, 1]}

        ######################
        # Output variables
        ######################
        # Populate the variables dictionary with any variables you want to compute
        # define intermediate variables and OCP function parameters
        c_i_s = [pybamm.surf(c_i[i]) for i in [0, 1]]
        x_i_s = [c_i_s[i] / c_i_max[i] for i in [0, 1]]
        i_0_i = [k_i[i] * F * (pybamm.sqrt(c_e) * pybamm.sqrt(c_i_s[i]) * pybamm.sqrt(c_i_max[i] - c_i_s[i])) for i in [0, 1]]
        eta_i = [2 * R * T / F * pybamm.arcsinh(j_i[i] * F / (2 * i_0_i[i])) for i in [0, 1]]
        U_i = [pybamm.FunctionParameter(f"{e.capitalize()} electrode OCP [V]", {"stoichiometry": x_i_s[i]}) for (i, e) in enumerate(electrodes)]

        # define output variables
        [U_n_plus_eta, U_p_plus_eta] = [U_i[i] + eta_i[i] for i in [0, 1]]
        V = U_p_plus_eta - U_n_plus_eta
        self.variables = {
            "Time [s]": pybamm.t,
            "Voltage [V]": V,
            "Current [A]": I,
            "Negative particle concentration [mol.m-3]": c_i[0],
            "Positive particle concentration [mol.m-3]": c_i[1],
            "Negative particle surface concentration [mol.m-3]": c_i_s[0],
            "Positive particle surface concentration [mol.m-3]": c_i_s[1],         
        }


    # The following attributes are used to define the default properties of the model,
    # which are used by the simulation class if unspecified.
    @property
    def default_geometry(self):
        return None

    @property
    def default_submesh_types(self):
        return None

    @property
    def default_var_pts(self):
        return None

    @property
    def default_spatial_methods(self):
        return None

    @property
    def default_solver(self):
        return None
    
    @property
    def default_quick_plot_variables(self):
        return None