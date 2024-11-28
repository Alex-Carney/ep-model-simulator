import streamlit as st
import numpy as np
import plotly.graph_objects as go
import symbolic_module as sm

# Setup symbolic equations and initial parameters
symbols_dict = sm.setup_symbolic_equations()


def compute_and_plot(params_list, lo_freqs, config_flags):
    """
    Compute photon numbers for multiple configurations and return a Plotly figure.

    Args:
        params_list: List of ModelParams objects for each configuration.
        lo_freqs: Array of LO frequencies.
        config_flags: Dictionary with keys as configuration names and values as booleans to toggle visibility.

    Returns:
        A Plotly figure with selected configurations.
    """
    fig = go.Figure()

    # Define configuration labels
    config_labels = {
        (1, 0): "Drive/Readout [1,0]",
        (1, 1): "Drive/Readout [1,1]",
        (0, 1): "Drive/Readout [0,1]",
    }

    # Iterate over each configuration
    for i, params in enumerate(params_list):
        config_key = tuple(params.drive_vector)
        if config_flags[config_key]:  # Only plot if the corresponding checkbox is selected
            ss_response = sm.get_steady_state_response_transmission(symbols_dict, params)
            photon_numbers = sm.compute_photon_numbers_transmission(ss_response, lo_freqs)

            # Add trace for the configuration
            fig.add_trace(go.Scatter(
                x=lo_freqs,
                y=np.log(photon_numbers),
                mode='lines',
                name=config_labels[config_key]
            ))

    # Update layout
    fig.update_layout(
        title="Photon Numbers vs LO Frequency",
        xaxis_title="LO Frequency (GHz)",
        yaxis_title="Log(Photon Numbers)",
        legend_title="Configurations",
        template="plotly_dark"
    )

    return fig


# Default values
default_params = sm.ModelParams(
    J_val=0.05,
    g_val=0,
    cavity_freq=6.0,
    w_y=5.9,
    gamma_vec=np.array([0.04, 0.04]),
    drive_vector=np.array([1, 0]),
    readout_vector=np.array([1, 0]),
    phi_val=2 * np.pi - 0.76,
)

# Streamlit sidebar for interactive parameter selection
st.sidebar.title("Model Parameters")

J_val = st.sidebar.slider("J_val", 0.001, 0.9, 0.05, step=0.001)
g_val = st.sidebar.slider("g_val", 0.0, 1.0, 0.0, step=0.1)
cavity_freq = st.sidebar.slider("cavity_freq", 5.7, 6.3, 6.0, step=0.001)
w_y = st.sidebar.slider("w_y", 5.7, 6.3, 6.0, step=0.001)
gamma_x = st.sidebar.slider("gamma_vec[0]", 0.001, 0.9, 0.04, step=0.001)
gamma_y = st.sidebar.slider("gamma_vec[1]", 0.001, 0.9, 0.04, step=0.001)
phi_val = st.sidebar.slider("phi_val", 0.0, 2 * np.pi, 2 * np.pi - 0.76, step=0.001)

# Drive/Readout configuration checkboxes
st.sidebar.title("Drive/Readout Configurations")
show_10 = st.sidebar.checkbox("Show [1,0]", value=True)
show_11 = st.sidebar.checkbox("Show [1,1]", value=True)
show_01 = st.sidebar.checkbox("Show [0,1]", value=False)

# Configuration flags
config_flags = {
    (1, 0): show_10,
    (1, 1): show_11,
    (0, 1): show_01,
}

# Update parameters based on sliders
params_10 = sm.ModelParams(
    J_val=J_val,
    g_val=g_val,
    cavity_freq=cavity_freq,
    w_y=w_y,
    gamma_vec=np.array([gamma_x, gamma_y]),
    drive_vector=np.array([1, 0]),
    readout_vector=np.array([1, 0]),
    phi_val=phi_val,
)

params_11 = sm.ModelParams(
    J_val=J_val,
    g_val=g_val,
    cavity_freq=cavity_freq,
    w_y=w_y,
    gamma_vec=np.array([gamma_x, gamma_y]),
    drive_vector=np.array([1, 1]),
    readout_vector=np.array([1, 1]),
    phi_val=phi_val,
)

params_01 = sm.ModelParams(
    J_val=J_val,
    g_val=g_val,
    cavity_freq=cavity_freq,
    w_y=w_y,
    gamma_vec=np.array([gamma_x, gamma_y]),
    drive_vector=np.array([0, 1]),
    readout_vector=np.array([0, 1]),
    phi_val=phi_val,
)

# Frequency range
lo_freqs = np.linspace(5.6, 6.4, 1000)

# List of configurations
params_list = [params_10, params_11, params_01]

# Compute eigenvalues and stability status
configs = [
    {'params': params_10, 'key': (1, 0), 'label': "Drive/Readout [1,0]", 'unstable': None},
    {'params': params_11, 'key': (1, 1), 'label': "Drive/Readout [1,1]", 'unstable': None},
    {'params': params_01, 'key': (0, 1), 'label': "Drive/Readout [0,1]", 'unstable': None},
]

selected_configs = []

for config in configs:
    key = config['key']
    if config_flags[key]:
        eigenvalues = sm.get_cavity_dynamics_eigenvalues_numeric(symbols_dict, config['params'])
        unstable = any(ev.real > 0 for ev in eigenvalues)
        config['unstable'] = unstable
        selected_configs.append(config)

# Display plot and computed values
st.title("EP Model Visualization")

# Compute additional values for display
kappa = gamma_x - gamma_y  # Difference between gamma_vec[0] and gamma_vec[1]
delta = cavity_freq - w_y  # Difference between cavity_freq and w_y

# Display computed values as markdown text
st.markdown(f"""
### Parameters:
- $\\kappa = {kappa:.2f} \\ \\mathrm{{MHz}}$
- $\\Delta = {delta:.2f} \\ \\mathrm{{GHz}}$
- $J = {J_val:.2f} \\ \\mathrm{{MHz}}$
- $\\phi = {phi_val:.2f}$
""")

# Display the plot
fig = compute_and_plot(params_list, lo_freqs, config_flags)
st.plotly_chart(fig)

# Display stability status
st.title("System Stability")

if len(selected_configs) > 0:
    cols = st.columns(len(selected_configs))

    for i, config in enumerate(selected_configs):
        with cols[i]:
            st.markdown(f"#### {config['label']}")
            if config['unstable']:
                st.markdown("<span style='color: red;'>🔴 **Unstable**</span>", unsafe_allow_html=True)
            else:
                st.markdown("<span style='color: green;'>🟢 **Stable**</span>", unsafe_allow_html=True)
