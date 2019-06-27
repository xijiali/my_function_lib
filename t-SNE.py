from __future__ import absolute_import
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

'''
Input:high dimensional feature [3368,2048] (Tensor) ; labels [3368]
Output: a figure
Name: the name of the figure
'''

def tsne(high_dimension_feature,labels,name=None):
    print('Begin to t-SNE')
    tsne_feature=TSNE(n_components=2, random_state=33).fit_transform(high_dimension_feature.detach().numpy())
    print('t-sne end')

    print('Begin to plot.')
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.scatter(tsne_feature[:, 0], tsne_feature[:, 1], c=labels, label=name)
    plt.savefig(name, dpi=120)
    plt.show()
    print('Plot end.')

