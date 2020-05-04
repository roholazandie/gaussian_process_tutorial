# import plotly.express as px
#
# df = px.data.gapminder()
#
# fig = px.bar(df, x="continent", y="pop", color="continent",
#   animation_frame="year", animation_group="country", range_y=[0, 4000000000])
# fig.show()

############################
import plotly.graph_objects as go
from gp_animation import GaussianProcessAnimation
import numpy as np
from kernels import rbf_kernel

n_dims = 150
n_frames = 100
x = np.linspace(-8, 8, n_dims).reshape(-1, 1)
kernel = rbf_kernel(x, x)
gaussian_process_animation = GaussianProcessAnimation(kernel, n_dims=n_dims, n_frames=n_frames)

traces = gaussian_process_animation.get_traces(3)

#fig = go.Figure()

#for trace in traces[0]:
fig = go.Figure(
    data=[go.Scatter(x=[0, 0], y=[0, 0])],
    layout=go.Layout(
        xaxis=dict(range=[-9, 9], autorange=False),
        yaxis=dict(range=[-20, 20], autorange=False),
        title="Start Title",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 50, "redraw": False},
                                "fromcurrent": True, "transition": {"duration": 10,
                                                                    "easing": "linear"}}],
                          )])]
    ),

    frames=[go.Frame(data=[go.Scatter(x=x, y=frame)]) for frame in traces],

  )





fig.show()