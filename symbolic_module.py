from dataclasses import dataclass
import numpy as np
import sympy as sp


@dataclass
class ModelSymbolics:
    J: sp.Symbol
    w_f: sp.Symbol
    w_y: sp.Symbol
    gam_y: sp.Symbol
    g: sp.Symbol
    w0: sp.MutableDenseMatrix
    gamma: sp.MutableDenseMatrix
    F: sp.MutableDenseMatrix
    phi_val: sp.Symbol
    steady_state_eqns: sp.Expr


@dataclass
class ModelParams:
    J_val: float
    g_val: float
    cavity_freq: float
    w_y: float
    gamma_vec: np.ndarray
    drive_vector: np.ndarray
    readout_vector: np.ndarray
    phi_val: float


def setup_symbolic_equations() -> ModelSymbolics:
    """
    Sets up the symbolic steady-state equations for the two-cavity system.
    Returns the symbolic variables and the steady-state equations.
    """
    # Symbolic variables
    J, w_f, w_y, gam_y, g = sp.symbols('J w_f w_y gam_y g', real=True)
    w_c1, w_c2 = sp.symbols('w_c1 w_c2', real=True)
    gam_1, gam_2 = sp.symbols('gamma_1 gamma_2', real=True)
    w0 = sp.Matrix([w_c1, w_c2])
    gamma = sp.Matrix([gam_1, gam_2])
    F1, F2 = sp.symbols('F1 F2')
    F = sp.Matrix([F1, F2])

    # Define the adjacency matrix with phase factor
    phi_val = sp.symbols('phi_val', real=True)
    cavity_adj_matrix = sp.Matrix([
        [0, sp.exp(1j * phi_val) * J],
        [J, 0]
    ])

    # Driving frequency vector
    num_cavities = cavity_adj_matrix.shape[0]
    wf = w_f * sp.ones(num_cavities, 1)

    # Define the cavity dynamics matrix
    cavity_dynamics_matrix = sp.zeros(num_cavities)
    cavity_dynamics_matrix[0, 0] = (cavity_adj_matrix[0, 0] * 1j
                                    - gamma[0] / 2
                                    - 1j * (w0[0] - wf[0]))
    cavity_dynamics_matrix[0, 1] = cavity_adj_matrix[0, 1] * 1j
    cavity_dynamics_matrix[1, 0] = cavity_adj_matrix[1, 0] * 1j
    cavity_dynamics_matrix[1, 1] = (cavity_adj_matrix[1, 1] * 1j
                                    - gamma[1] / 2
                                    - 1j * (w0[1] - wf[1]))

    # Steady-state equations
    steady_state_eqns = cavity_dynamics_matrix.inv() * F
    steady_state_eqns_simplified = sp.simplify(steady_state_eqns)

    return ModelSymbolics(J, w_f, w_y, gam_y, g, w0, gamma, F, phi_val, steady_state_eqns_simplified)


def get_steady_state_response_NR(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    Returns a function that computes the steady-state response for the non-PT symmetric case.
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for non-PT symmetric case
    substitutions = {
        symbols_dict.w0[0]: params.cavity_freq,
        symbols_dict.w0[1]: symbols_dict.w_y,  # Keep w_y symbolic
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: params.gamma_vec[1],
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
              params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with w_y and w_f as variables
    return sp.lambdify((symbols_dict.w_y, symbols_dict.w_f), ss_eqn, 'numpy')


def get_steady_state_response_PT(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    Returns a function that computes the steady-state response for the PT symmetric case.
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for PT symmetric case
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: symbols_dict.gam_y,  # Keep gam_y symbolic
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
              params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with gam_y and w_f as variables
    return sp.lambdify((symbols_dict.gam_y, symbols_dict.w_f), ss_eqn, 'numpy')


def get_steady_state_response_transmission(symbols_dict: ModelSymbolics, params: ModelParams) -> sp.Expr:
    """
    FOR TRANSMISSION, ALL PARAMETERS MUST BE FILLED. The only unfilled parameter is w_f
    """
    # Unpack symbols
    w0 = symbols_dict.w0
    gamma = symbols_dict.gamma
    F = symbols_dict.F
    steady_state_eqns = symbols_dict.steady_state_eqns

    # Substitutions for PT symmetric case
    substitutions = {
        w0[0]: params.cavity_freq,
        w0[1]: params.w_y,
        symbols_dict.J: params.J_val,
        symbols_dict.g: params.g_val,
        F[0]: params.drive_vector[0],
        F[1]: params.drive_vector[1],
        gamma[0]: params.gamma_vec[0],
        gamma[1]: params.gamma_vec[1],
        symbols_dict.phi_val: params.phi_val
    }

    ss_eqns_instantiated = steady_state_eqns.subs(substitutions)
    ss_eqn = (params.readout_vector[0] * ss_eqns_instantiated[0] +
              params.readout_vector[1] * ss_eqns_instantiated[1])

    # Lambdify with gam_y and w_f as variables
    return sp.lambdify(symbols_dict.w_f, ss_eqn, 'numpy')


def compute_photon_numbers_NR(ss_response_func, w_y_vals, w_f_vals):
    """
    Computes the photon numbers for the non-PT symmetric case.
    ss_response_func: steady-state response function from get_steady_state_response_non_PT
    w_y_vals: array of YIG frequencies
    w_f_vals: array of LO frequencies
    Returns a 2D array of photon numbers.
    """
    W_Y, W_F = np.meshgrid(w_y_vals, w_f_vals, indexing='ij')
    photon_numbers_complex = ss_response_func(W_Y, W_F)
    photon_numbers_real = np.abs(photon_numbers_complex) ** 2
    return photon_numbers_real


def compute_photon_numbers_PT(ss_response_func, gam_y_vals, w_f_vals):
    """
    Computes the photon numbers for the PT symmetric case.
    ss_response_func: steady-state response function from get_steady_state_response_PT
    gam_y_vals: array of gamma_y values
    w_f_vals: array of LO frequencies
    Returns a 2D array of photon numbers.
    """
    GAM_Y, W_F = np.meshgrid(gam_y_vals, w_f_vals, indexing='ij')
    photon_numbers_complex = ss_response_func(GAM_Y, W_F)
    photon_numbers_real = np.abs(photon_numbers_complex) ** 2
    return photon_numbers_real


def compute_photon_numbers_transmission(ss_response_func, w_f_vals):
    """
    Computes the photon numbers for the transmission case.
    ss_response_func: steady-state response function from get_st
    w_f_vals: array of LO frequencies
    Returns a 2D array of photon numbers.
    """
    photon_numbers_complex = ss_response_func(w_f_vals)
    photon_numbers_real = np.abs(photon_numbers_complex) ** 2
    return photon_numbers_real


# Example usage
if __name__ == "__main__":
    # Setup symbolic equations
    symbols_dict = setup_symbolic_equations()

    # Define parameters
    params = {
        'J_val': 0.06,
        'g_val': 0.025 - 0.04,
        'cavity_freq': 6.0,  # GHz
        'w_y': sp.symbols('w_y'),  # Keep symbolic for lambdify
        'gamma_vec': np.array([0.025, 0.04]),
        'drive_vector': np.array([1, 0]),
        'readout_vector': np.array([1, 0]),
        'phi_val': np.pi - 0.1,
    }

    # Non-PT symmetric case
    ss_response_non_PT = get_steady_state_response_NR(symbols_dict, params)

    # Define frequency ranges
    yig_freqs = np.linspace(5.6, 6.4, 1000)  # YIG frequencies in GHz
    lo_freqs = np.linspace(5.6, 6.4, 1000)  # LO frequencies in GHz

    # Compute photon numbers for non-PT symmetric case
    photon_numbers_non_PT = compute_photon_numbers_NR(ss_response_non_PT, yig_freqs, lo_freqs)

    # PT symmetric case
    params_PT = params.copy()
    ss_response_PT = get_steady_state_response_PT(symbols_dict, params_PT)

    # Define gamma_y values for PT symmetric case
    gamma_y_vals = np.linspace(0.001, 0.05, 1000)

    # Compute photon numbers for PT symmetric case
    photon_numbers_PT = compute_photon_numbers_PT(ss_response_PT, gamma_y_vals, lo_freqs)

    # At this point, you have the photon_numbers_non_PT and photon_numbers_PT arrays.
    # You can proceed with plotting or further analysis as needed.
