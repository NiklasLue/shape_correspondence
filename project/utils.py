import torch
import open3d as o3d
from dpfm.utils import FrobeniusLoss

def farthest_point_sample(xyz, ratio):
    xyz = xyz.t().unsqueeze(0)
    npoint = int(ratio * xyz.shape[1])
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids[0]


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def read_ply(path):
    mesh = o3d.io.read_triangle_mesh(path)
    # vertices = np.asarray(mesh.vertices)
    # faces = np.asarray(mesh.triangles, dtype=int)

    return mesh.vertices, mesh.triangles
    
class DPFMLoss_unsup(nn.Module):
    def __init__(self, w_orth=1, w_bij=1):
        super().__init__()

        self.w_orth = w_orth
        self.w_bij = w_bij

        self.frob_loss = FrobeniusLoss()


    def forward(self, C12, C21, map21, feat1, feat2):
        loss = 0
        I = 0

        # orthogonality loss
        orth_loss = self.frob_loss(torch.bmm(C12, C12.transpose(1, 2)), I) * self.w_orth         
        orth_loss += self.frob_loss(torch.bmm(C21, C21.transpose(1, 2)), I) * self.w_orth
        loss += orth_loss

        # bijectivity loss
        bij_loss = self.frob_loss(torch.bmm(C12, C21), I) * self.w_bij
        loss += bij_loss


        return loss