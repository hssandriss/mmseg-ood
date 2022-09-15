import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import plotly.express as px
import torch
# with open('initial_1000_z_samples_base_I.npy', 'rb') as f:
#     z_ = np.load(f)
# np.savetxt("initial_1000_z_samples_base_I.csv", z_)
# import ipdb; ipdb.set_trace()
data_path_initial = 'initial_1000_z_samples_base.csv'
data_path_initial_I = 'initial_5000_z_samples_base_I.csv'
data_path_trained2xradial = 'trained_5000_z_samples_2xradial.csv'
data_path_trained2xplanar = 'trained_1000_z_samples_2xplanar.csv'
data_path_trained1xplanar = 'trained_1000_z_samples_1xplanar.csv'
# data_path_trained_1xplanar =
with open(data_path_initial_I, 'r') as f:
    data_initial_I = np.loadtxt(f)

with open(data_path_initial, 'r') as f:
    data_initial = np.loadtxt(f)

with open(data_path_trained2xradial, 'r') as f:
    data_trained2xradial = np.loadtxt(f)


with open(data_path_trained2xplanar, 'r') as f:
    data_trained2xplanar = np.loadtxt(f)

with open(data_path_trained1xplanar, 'r') as f:
    data_trained1xplanar = np.loadtxt(f)
pca = PCA(n_components=4)
all_data = np.concatenate((data_initial_I,
                           data_initial,
                           data_trained2xradial,
                           data_trained2xplanar,
                           data_trained1xplanar), axis=0)
color = np.array(
    ["base_I" for _ in range(5000)] +
    ["base_MAP" for _ in range(1000)] +
    ["trained_2xradial" for _ in range(5000)] +
    ["trained_2xplanar" for _ in range(1000)] +
    ["trained_1xplanar" for _ in range(1000)]
)
projections = pca.fit_transform(all_data)

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    projections,
    labels=labels,
    dimensions=range(4),
    title=f"PCA(4 components)",
    color=color
)
fig.update_traces(diagonal_visible=False)
fig.write_html(f'pca_4cp_planar1xplanar2xbase_radial2xbaseI5000.html')
