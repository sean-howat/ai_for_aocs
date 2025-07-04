import numpy as np
import plotly.graph_objects as go
from astropy.time import Time
from astropy import units as u
from poliastro.bodies import Earth, Mars, Jupiter, Saturn, Venus, Sun
from poliastro.twobody import Orbit
from poliastro.ephem import Ephem

# --- Load data ---
#data = np.load("trajectory_data_rllib.npz") 
data = np.load("trajectory_data_sb3.npz")

observations = data["observations"]
MAX_POS = 4e8  # must match environment normalization important for scaling!, todo add this to json file
sc_pos = np.array([obs[3:6] * MAX_POS for obs in observations])
dvs = data["dvs"]
jd_times = data["jd_times"]
num_frames = int(data["num_frames"])
epochs = Time(jd_times, format="jd", scale="tdb")

# --- Planet positions ---
def get_positions(body):
    return [Orbit.from_ephem(Sun, Ephem.from_body(body, t), epoch=t).r.to(u.km).value for t in epochs]

planet_positions = {
    "VENUS": get_positions(Venus),
    "EARTH": get_positions(Earth),  
    "MARS": get_positions(Mars),
    "JUPITER": get_positions(Jupiter),
    "SATURN": get_positions(Saturn),
}

# --- Flyby radii --- not used really,
def compute_flyby_radii(epoch):
    planet_names = ["VENUS", "EARTH", "MARS", "JUPITER", "SATURN"]
    flyby_radii = {}
    for name in planet_names:
        body = eval(name.title())
        orbit = Orbit.from_ephem(Sun, Ephem.from_body(body, epoch), epoch=epoch)
        a = orbit.a.to(u.km)
        m = body.k.to(u.km**3 / u.s**2)
        M = Sun.k.to(u.km**3 / u.s**2)
        soi = a * (m / M) ** (2 / 5)
        flyby_radii[name] = soi
    return flyby_radii

flyby_radii = compute_flyby_radii(epochs[0])

traj_x, traj_y, traj_z = sc_pos[:, 0], sc_pos[:, 1], sc_pos[:, 2]

# --- deltav Flip Markers --- doesnt work, to remove
dv_flips = []
for i in range(1, len(dvs)):
    if np.dot(dvs[i], dvs[i - 1]) < 0:  # Flip in direction
        dv_flips.append(sc_pos[i])

# --- Flyby Arcs --- 
flyby_arcs = []
for name, positions in planet_positions.items():
    soi = flyby_radii[name].to_value(u.km)
    for i in range(num_frames):
        planet_r = np.array(positions[i])
        spacecraft_r = sc_pos[i]
        distance = np.linalg.norm(planet_r - spacecraft_r)
        if distance < soi:
            flyby_arcs.append(go.Scatter3d(
                x=[planet_r[0], spacecraft_r[0]],
                y=[planet_r[1], spacecraft_r[1]],
                z=[planet_r[2], spacecraft_r[2]],
                mode='lines',
                line=dict(color='magenta', width=3, dash='dot'),
                name=f'Flyby arc ({name})',
                showlegend=False
            ))

# --- trajecotry data + plot stuff
init_data = []
for name in planet_positions:
    x, y, z = planet_positions[name][0]
    init_data.append(go.Scatter3d(
        x=[x], y=[y], z=[z],
        mode='markers+text',
        text=[name],
        textposition='top center',
        marker=dict(size=6),
        name=name
    ))

init_data.append(go.Scatter3d(
    x=[traj_x[0]], y=[traj_y[0]], z=[traj_z[0]],
    mode='markers',
    marker=dict(size=6, color='cyan'),
    name='Spacecraft'
))

init_data.append(go.Scatter3d(
    x=traj_x, y=traj_y, z=traj_z,
    mode='lines',
    line=dict(color='cyan', width=2, dash='dot'),
    name='Trajectory'
))

init_data.append(go.Scatter3d(
    x=[0], y=[0], z=[0],
    mode='markers+text',
    text=['SUN'],
    textposition='top center',
    marker=dict(size=10, color='yellow'),
    name='Sun'
))

# 0.5 AU no fly zone sphere---
au_km = (1 * u.AU).to_value(u.km)
radius = 0.5 * au_km  # 0.5 AU in km

phi, theta = np.mgrid[0:np.pi:20j, 0:2*np.pi:40j]
x_sphere = radius * np.sin(phi) * np.cos(theta)
y_sphere = radius * np.sin(phi) * np.sin(theta)
z_sphere = radius * np.cos(phi)

x_flat = x_sphere.flatten()
y_flat = y_sphere.flatten()
z_flat = z_sphere.flatten()

i, j, k = [], [], []
rows, cols = x_sphere.shape
for r in range(rows - 1):
    for c in range(cols - 1):
        idx = r * cols + c
        i += [idx, idx + 1]
        j += [idx + cols, idx + cols + 1]
        k += [idx + 1, idx + cols]

init_data.append(go.Mesh3d(
    x=x_flat, y=y_flat, z=z_flat,
    i=i, j=j, k=k,
    opacity=0.1,
    color='lightblue',
    name='0.5 AU Zone',
    showscale=False
))



# Add flyby arcs
init_data.extend(flyby_arcs)

# Frames for Animation 
frames = []
for i in range(num_frames):
    frame_data = []

    for name in planet_positions:
        x, y, z = planet_positions[name][i]
        frame_data.append(go.Scatter3d(
            x=[x], y=[y], z=[z],
            mode='markers+text',
            text=[name],
            textposition='top center',
            marker=dict(size=6),
            name=name,
            showlegend=False
        ))

    frame_data.append(go.Scatter3d(
        x=[traj_x[i]], y=[traj_y[i]], z=[traj_z[i]],
        mode='markers',
        marker=dict(size=6, color='cyan'),
        name='Spacecraft',
        showlegend=False
    ))

    frame_data.append(go.Scatter3d(
        x=traj_x[:i+1], y=traj_y[:i+1], z=traj_z[:i+1],
        mode='lines',
        line=dict(color='cyan', width=2, dash='dot'),
        name='Trajectory',
        showlegend=False
    ))

    frame_data.append(go.Scatter3d(
        x=[0], y=[0], z=[0],
        mode='markers+text',
        text=['SUN'],
        textposition='top center',
        marker=dict(size=10, color='yellow'),
        name='Sun',
        showlegend=False
    ))
 


    frames.append(go.Frame(data=frame_data, name=f"Frame {i}"))

# Use fixed 2e9 km for Full zoomm, to fix different zoom levels, inner zoom too small
max_zoom = 2.0e9

zoom_levels = {
    "Full": max_zoom,
    "Outer Planets": 1.0e9,
    "Inner Planets": 3.0e8,
    "Earth Vicinity": 1.0e8
}
# Get Saturn's final position (last frame), todo change to mars
saturn_final_r = np.array(planet_positions["SATURN"][-1])  # shape: (3,)
init_data.append(go.Scatter3d(
    x=[saturn_final_r[0]],
    y=[saturn_final_r[1]],
    z=[saturn_final_r[2]],
    mode='markers+text',
    text=["Saturn Final"],
    textposition='bottom center',
    marker=dict(size=7, color='orange', symbol='circle'),
    name='Saturn Final'
))

fig = go.Figure(
    data=init_data,
    layout=go.Layout(
        title=" RL Gravity Assist with Δv Direction Flip Markers",
        scene=dict(
            xaxis=dict(range=[-max_zoom, max_zoom], title='X (km)'),
            yaxis=dict(range=[-max_zoom, max_zoom], title='Y (km)'),
            zaxis=dict(range=[-max_zoom, max_zoom], title='Z (km)'),
            dragmode="orbit",  
        ),

        updatemenus=[
            dict(type='buttons', showactive=True, buttons=[
                dict(label=' Play', method='animate',
                     args=[None, dict(frame=dict(duration=60, redraw=True), fromcurrent=True)])
            ], x=0.1, y=1.15),
            dict(type="dropdown", showactive=True, buttons=[
                dict(label=label, method="relayout", args=[{
                    "scene.xaxis.range": [-r, r],
                    "scene.yaxis.range": [-r, r],
                    "scene.zaxis.range": [-r, r],
                }]) for label, r in zoom_levels.items()
            ], x=0.6, y=1.15, xanchor='left', yanchor='top')
        ],
        margin=dict(l=0, r=0, t=50, b=0)
    ),
    frames=frames
)

fig.write_html("rl_gravity_assist_with_dv_flips.html")
print(" Visualization saved to rl_gravity_assist_with_dv_flips.html")

# for some reason wsl doesnt like import webbrowser, i simply use ctrl+ O on browser, 

