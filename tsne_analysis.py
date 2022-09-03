import numpy as np
from sklearn.manifold import TSNE
import plotly.express as px
data_path = '1000_z_samples_2xradial.npy'
with open(data_path, 'rb') as f:
    data = np.load(f)
suffixe = data_path.split('.')[0]

tsne = TSNE(n_components=2)
projections = tsne.fit_transform(data)

fig = px.scatter(
    projections, x=0, y=1, title=f"TSNE(2D): {data_path}"

)
fig.write_html(f'tsne_2d_{suffixe}.html')
