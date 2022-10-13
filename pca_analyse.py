import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
import joblib

initial_samples = joblib.load('base_dist.pkl')
flow_samples_4radial_2fc_relu = joblib.load('flow_dist_4radial+2xfc+relu.pkl')
w_samples_4radial_2fc_relu = joblib.load('proj_w_4radial+2xfc+relu.pkl')

flow_samples_4sylv_1fc = joblib.load('flow_dist_4sylv+1xfc.pkl')
w_samples_4sylv_1fc = joblib.load('proj_w_4sylv+1xfc.pkl')

transformed_initial_flow = PCA(n_components=4).fit_transform(np.concatenate(
    (initial_samples, flow_samples_4radial_2fc_relu, flow_samples_4sylv_1fc), axis=0))

pca = PCA(n_components=4)
w_samples = pca.fit_transform(np.concatenate((w_samples_4radial_2fc_relu, w_samples_4sylv_1fc), axis=0))
all_samples = np.concatenate((transformed_initial_flow, w_samples), axis=0)

color = np.array(
    ["Base Distribution" for _ in range(5000)] +
    ["Flow Distribution (4radial+fc+relu+fc)" for _ in range(5000)] +
    ["Flow Distribution (4sylv+1xfc)" for _ in range(5000)] +
    ["Proj Weights (4radial+fc+relu+fc)" for _ in range(5000)] +
    ["Proj Weights (4sylv+1xfc)" for _ in range(5000)]
)
labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    all_samples,
    labels=labels,
    dimensions=range(4),
    title=f"PCA(4 components)",
    color=color
)

fig.update_traces(diagonal_visible=False, marker=dict(size=3, opacity=0.3))
fig.write_html(f'pca_4cp.html')
