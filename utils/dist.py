import torch

def eucl_dist(x, y):
    """ Compute Pairwise (Squared Euclidean) Distance

    Input:
        x: embedding of size M x D
        y: embedding of size N x D

    Output:
        dist: pairwise distance of size M x N
    """
    x2 = torch.sum(x**2, dim=1, keepdim=True).expand(-1, y.size(0))
    y2 = torch.sum(y**2, dim=1, keepdim=True).t().expand(x.size(0), -1)
    xy = x.mm(y.t())
    return x2 - 2*xy + y2


def rbf_graph(x, y, sigma):
    diff = eucl_dist(x, y)
    g = torch.exp(diff / (-2.0 * sigma**2))
    return g / torch.sum(g, dim=1, keepdim=True)


def cosine_dist(x, y):
    """ Compute consin dist

    Input:
        x: embedding of size M x D
        y: embedding of size N x D

    Output:
        dist: pairwise distance of size M x N
    """
    xy = x.mm(y.t())
    x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
    y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
    xy_norm = x_norm.mm( y_norm.t() )
    return xy / xy_norm.add(1e-10)


def neighbor_graph(x):
    neighbor_n = x.size(0)
    x1 = x.unsqueeze(0).expand(neighbor_n, -1)
    x2 = x.unsqueeze(1).expand(-1, neighbor_n)
    return x1.eq(x2).float()
