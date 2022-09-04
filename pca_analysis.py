import numpy as np
from sklearn.decomposition import PCA
import plotly.express as px
data_path = '1000_z_samples_2xplanar.npy'

with open(data_path, 'rb') as f:
    data = np.load(f)

suffixe = data_path.split('.')[0]

pca = PCA(n_components=4)
projections = pca.fit_transform(data)

labels = {
    str(i): f"PC {i+1} ({var:.1f}%)"
    for i, var in enumerate(pca.explained_variance_ratio_ * 100)
}

fig = px.scatter_matrix(
    projections,
    labels=labels,
    dimensions=range(4),
    title=f"PCA(4 components): {data_path}"
)
fig.update_traces(diagonal_visible=False)
fig.write_html(f'pca_4cp_{suffixe}.html')
