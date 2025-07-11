from tqdm import tqdm
import torch
from torchvision import models
import torch.nn.functional as F

from pairs import pair
from test import precision


def KL(alpha, c):
    beta = torch.ones((1, c)).cuda()
    # beta = torch.ones((1, c))
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1, keepdim=True) - torch.lgamma(S_beta)
    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)
    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1, keepdim=True) + lnB + lnB_uni
    return kl


def ce_loss(p, alpha, c, global_step, annealing_step):
    S = torch.sum(alpha, dim=1, keepdim=True)
    E = alpha - 1
    label = F.one_hot(p, num_classes=c)
    print(label.shape,S.shape,alpha.shape)
    A = torch.sum(label * (torch.digamma(S) - torch.digamma(alpha)), dim=1, keepdim=True)
    annealing_coef = min(1, global_step / annealing_step)
    alp = E * (1 - label) + 1
    B = annealing_coef * KL(alp, c)
    return (A + B)


def DS_Combin(alpha, classes=10):
    """
    :param alpha: All Dirichlet distribution parameters.
    :return: Combined Dirichlet distribution parameters.
    """
    def DS_Combin_two(alpha1, alpha2):
        """
        Dempster’s  combination  rule  for  two  independent  sets  of  masses
        :param alpha1: Dirichlet distribution parameters of view 1
        :param alpha2: Dirichlet distribution parameters of view 2
        :return: Combined Dirichlet distribution parameters
        """
        alpha = dict()
        alpha[0], alpha[1] = alpha1, alpha2
        b, S, E, u = dict(), dict(), dict(), dict()
        for v in range(2):
            S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
            E[v] = alpha[v]-1
            b[v] = E[v]/(S[v].expand(E[v].shape))
            u[v] = classes/S[v]

        # b^0 @ b^(0+1)
        bb = torch.bmm(b[0].view(-1, classes, 1), b[1].view(-1, 1, classes))
        # b^0 * u^1
        uv1_expand = u[1].expand(b[0].shape)
        bu = torch.mul(b[0], uv1_expand)
        # b^1 * u^0
        uv_expand = u[0].expand(b[0].shape)
        ub = torch.mul(b[1], uv_expand)
        # calculate C
        bb_sum = torch.sum(bb, dim=(1, 2), out=None)
        bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
        # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
        C = bb_sum - bb_diag

        # calculate b^a
        b_a = (torch.mul(b[0], b[1]) + bu + ub)/((1-C).view(-1, 1).expand(b[0].shape))
        # calculate u^a
        u_a = torch.mul(u[0], u[1])/((1-C).view(-1, 1).expand(u[0].shape))
        # test = torch.sum(b_a, dim = 1, keepdim = True) + u_a #Verify programming errors

        # calculate new S
        S_a = classes / u_a
        # calculate new e_k
        e_a = torch.mul(b_a, S_a.expand(b_a.shape))
        alpha_a = e_a + 1
        return alpha_a

    for v in range(len(alpha)-1):
        if v==0:
            alpha_a = DS_Combin_two(alpha[0], alpha[1])
        else:
            alpha_a = DS_Combin_two(alpha_a, alpha[v+1])
    return alpha_a


def get_mik(data_loader, args):
    resnet18 = models.resnet18(pretrained=True)
    resnet18.to(args.device)
    resnet18.eval()

    features1, features2, features3 = [], [], []
    labels = []
    for inputs1, inputs2, inputs3, label in tqdm(data_loader):
        inputs1, inputs2, inputs3, label = (inputs1.to(args.device), inputs2.to(args.device),
                                            inputs3.to(args.device), label.to(args.device))
        with torch.no_grad():
            outputs1 = resnet18(inputs1)
            outputs2 = resnet18(inputs2)
            outputs3 = resnet18(inputs3)
        features1.append(outputs1)
        features2.append(outputs2)
        features3.append(outputs3)
        labels.append(label)

    features1 = torch.cat(features1).cpu().numpy()
    features2 = torch.cat(features2).cpu().numpy()
    features3 = torch.cat(features3).cpu().numpy()
    features = [features1, features2, features3]
    labels = torch.cat(labels).cpu().numpy()

    mi_dict = {}
    for i in range(len(features)):
        # me = ['cityblock', 'cosine', 'euclidean', 'l1', 'l2', 'manhattan']
        me = ['cityblock']
        mi_idx_all = []
        error_mi_all = []
        for met in me:
            g, e, mi_idx, error_mi = pair(args.mi, args.k, features[i], labels, metrix=met)
            mi_idx_all.append(mi_idx)
            error_mi_all.append(error_mi)
            print(f'view{i} met is: {met}, error_mi is: {error_mi}')
        mi_dict[i] = mi_idx_all[error_mi_all.index(min(error_mi_all))]

    return mi_dict
