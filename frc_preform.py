import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.set_page_config(layout="wide", page_title="FRC Collagen Preform Generator")
st.title("Interactive Collagen Fiber Network ")

# -- Sidebar Controls --
st.sidebar.header("Preform Parameters")
num_layers = st.sidebar.number_input("Number of Layers", min_value=1, max_value=10, value=4)
fibers_per_layer = st.sidebar.slider("Fibrils per Layer", min_value=10, max_value=300, value=75)

st.sidebar.subheader("In-Plane Orientation")
layer_rotation = st.sidebar.slider("Inter-layer Rotation (°)", 0, 90, 45)
dispersion = st.sidebar.slider("In-plane Dispersion Variance (°)", 0.0, 45.0, 10.0)

st.sidebar.subheader("3D Architecture (Z-Direction)")
z_angle_max = st.sidebar.slider("Out-of-Plane Z-Angle (°)", 0.0, 45.0, 8.0, 
                                help="Tilts fibers up/down out of the XY plane to tangle with adjacent layers.")

st.sidebar.subheader("Micromechanics")
crimp_amp = st.sidebar.slider("Crimp Amplitude", 0.0, 2.0, 0.4)
crimp_freq = st.sidebar.slider("Crimp Frequency", 0.5, 5.0, 1.5)

# -- Geometry Generation --
fig = go.Figure()
colors = ['#1f77b4', '#F2E10A', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
bounding_box = 10.0 

# 1. Generate Standard & Z-Angled Layers
for i in range(num_layers):
    base_angle = np.radians(i * layer_rotation)
    z_offset = i * 1.5
    layer_color = colors[i % len(colors)]

    for f in range(fibers_per_layer):
        # Yaw (In-plane rotation)
        angle_z = base_angle + np.radians(np.random.normal(0, dispersion))
        # Pitch (Out-of-plane rotation)
        angle_y = np.radians(np.random.normal(0, z_angle_max))
        
        cx = np.random.uniform(-bounding_box/2, bounding_box/2)
        cy = np.random.uniform(-bounding_box/2, bounding_box/2)
        phase = np.random.uniform(0, 2 * np.pi)

        t = np.linspace(-bounding_box, bounding_box, 150)
        x_base = t
        y_base = crimp_amp * np.sin(crimp_freq * t + phase)
        z_base = (crimp_amp * 0.4) * np.cos(crimp_freq * t + phase)

        # Apply Pitch (tilt into Z-axis)
        x_pitch = x_base * np.cos(angle_y) + z_base * np.sin(angle_y)
        y_pitch = y_base
        z_pitch = -x_base * np.sin(angle_y) + z_base * np.cos(angle_y)

        # Apply Yaw (rotate in XY plane) and translate
        X = x_pitch * np.cos(angle_z) - y_pitch * np.sin(angle_z) + cx
        Y = x_pitch * np.sin(angle_z) + y_pitch * np.cos(angle_z) + cy
        Z = z_pitch + z_offset

        mask = (X >= -bounding_box/2) & (X <= bounding_box/2) & (Y >= -bounding_box/2) & (Y <= bounding_box/2)
        
        if np.any(mask):
            fig.add_trace(go.Scatter3d(
                x=X[mask], y=Y[mask], z=Z[mask],
                mode='lines',
                line=dict(color=layer_color, width=4),
                showlegend=False, hoverinfo='skip'
            ))



# -- Render UI --
fig.update_layout(
    scene=dict(
        xaxis=dict(range=[-bounding_box/2, bounding_box/2], visible=False),
        yaxis=dict(range=[-bounding_box/2, bounding_box/2], visible=False),
        zaxis=dict(range=[-2, num_layers*1.5 + 2], visible=False),
        bgcolor='rgb(20, 20, 20)' 
    ),
    margin=dict(l=0, r=0, b=0, t=0),
    height=750
)

st.plotly_chart(fig, width='stretch')
