import torch
import torch.nn as nn
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
    
def get_rank(evals1, evals2):
    est_rank = 0
    m = max(evals1)
    for i in range(len(evals2)):
        if evals2[i] < m:
            est_rank += 1
    return est_rank



class DPFMLoss_unsup(nn.Module):
    def __init__(self, w_orth_C1=1, w_orth_C2=0.01, w_bij=1, w_diff = 0.1):
        super().__init__()

        self.w_bij = w_bij
        self.w_orth_C1 = w_orth_C1
        self.w_orth_C2 = w_orth_C2
        self.w_diff = w_diff
        self.frob_loss = FrobeniusLoss()
        # self.orth_loss_C1 = OrthLoss_C1()
        # self.orth_loss_C2 = OrthLoss_C2()
        # self.bij_loss = BijLoss()
        # self.diff_loss = DiffLoss()


    def forward(self, C12, C21, evals1, evals2):
        device = C12.device
        I = torch.empty(C12.size()).to(device)
        for i in range(I.size(dim=0)):
            J = torch.eye(I.size(dim=1))
            r = get_rank(evals1[i,:], evals2[i,:])
            for k in range(r, I.size(dim=1)):
                J[k,k] = 0
            I[i] = J
            
        loss = 0
        

        # orthogonality loss C1
        orth_loss_C1 = self.frob_loss(torch.bmm(C12, C12.transpose(1, 2)), I)
        # orth_loss_C1 = self.orth_loss_C1(C12, I)
        if (self.w_orth_C1 > 0):
            loss += self.w_orth_C1 * orth_loss_C1
        
        # orthogonality loss C1
        orth_loss_C2 = self.frob_loss(torch.bmm(C21.transpose(1, 2), C21), I)
        # orth_loss_C2 = self.orth_loss_C2(C21, I)
        if (self.w_orth_C2 > 0):
            loss += self.w_orth_C2 * orth_loss_C2

        # bijectivity loss
        bij_loss = self.frob_loss(torch.bmm(C12, C21), I)
        # bij_loss = self.bij_loss(C12, C21, I)
        if (self.w_bij > 0):
            loss += self.w_bij * bij_loss
        
        # difference loss
        diff_loss = self.frob_loss(C12, C21.transpose(1, 2))
        # diff_loss = self.diff_loss(C12, C21)
        if (self.w_diff > 0):
            loss += self.w_diff * diff_loss

        return loss, orth_loss_C1, orth_loss_C2, bij_loss
        
        
# class OrthLoss_C1(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.frob_loss = FrobeniusLoss()


#     def forward(self, C12, I_r): 

#         # orthogonality loss
#         orth_loss = self.frob_loss(torch.bmm(C12, C12.transpose(1, 2)), I_r)


#         return orth_loss
        
# class OrthLoss_C2(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.frob_loss = FrobeniusLoss()


#     def forward(self, C21, I_r): 

#         # orthogonality loss
#         orth_loss = self.frob_loss(torch.bmm(C21.transpose(1, 2), C21), I_r)


#         return orth_loss       
        
# class BijLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.frob_loss = FrobeniusLoss()


#     def forward(self, C12, C21, I_r): 

#         # bijectivity loss
#         bij_loss = self.frob_loss(torch.bmm(C12, C21), I_r)


#         return bij_loss
        
# class DiffLoss(nn.Module):
#     def __init__(self):
#         super().__init__()

#         self.frob_loss = FrobeniusLoss()


#     def forward(self, C12, C21): 

#         # bijectivity loss
#         diff_loss = self.frob_loss(C12, C21.transpose(1, 2))


#         return diff_loss   