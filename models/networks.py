import torch
import torch.nn as nn
import torch.sparse
from torch.autograd import Variable
import numpy as np
import sys
from torch.autograd import Function
import math
import h5py
import json
# from . import resnet1
import matplotlib.pyplot as plt
from skimage.transform import resize
from gaussian import GaussianPyramid

###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type):
    if norm_type == 'batch':
        norm_layer = nn.BatchNorm2d
    elif norm_type == 'instance':
        norm_layer = nn.InstanceNorm2d
    else:
        print('normalization layer [%s] is not found' % norm)
    return norm_layer

def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = MultiUnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    else:
        print('Generator model name [%s] is not recognized' % which_model_netG)

    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])

    netG.apply(weights_init)

    return netG

def define_HumanJudgement(input_nc, which_model):
    if which_model == 'simple':
        return nn.Conv2d(input_nc, 1, kernel_size=1)
    elif which_model == 'mlp':
        return nn.Sequential(nn.Conv2d(input_nc, 8, kernel_size=1), nn.ReLU(False), nn.Conv2d(8, 8, kernel_size=1), nn.ReLU(False), nn.Conv2d(8, 1, kernel_size=1))
    elif which_model == 'residual':
        model = nn.Sequential(nn.Conv2d(input_nc, 8, kernel_size=1), nn.ReLU(False), nn.Conv2d(8, 8, kernel_size=1), nn.ReLU(False), nn.Conv2d(8, 1, kernel_size=1), nn.Tanh())
        return ResidualNetwork([1], model)

def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        print('Discriminator model name [%s] is not recognized' %
              which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################

class Sparse(Function):
    # Sparse matrix for S
    def forward(self, input, S):
        self.save_for_backward(S)
        output = torch.mm(S, input)
        # output = output.cuda()
        return output

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):
        S,  = self.saved_tensors
        grad_weight  = None
        grad_input = torch.mm(S.t(), grad_output)
        # grad_input = grad_input.cuda()
        return grad_input, grad_weight

class ResidualNetwork(nn.Module):
    def __init__(self, pass_selection, adjust_model):
        super(ResidualNetwork, self).__init__()
        self.pass_selection = pass_selection
        self.adjust_model = adjust_model

    def forward(self, x):
        passforward = torch.index_select(x, 1, torch.cuda.LongTensor(self.pass_selection))
        return self.adjust_model(x) + passforward


def get_human_pair_classifier(in_dim, model_type, bilinear, num_layers, inner_dim):
    if model_type == 'ternary':
        return HumanPairClassifier(in_dim, model_type, bilinear, num_layers, inner_dim)
    elif model_type == 'binary':
        return HumanPairClassifier(in_dim, model_type, bilinear, num_layers, inner_dim)
    elif model_type == 'single_score':
        return SingleScoreHumanClassifier(in_dim, model_type, False, bilinear, num_layers, inner_dim)
    elif model_type == 'single_score_const_thresh':
        return SingleScoreHumanClassifier(in_dim, 'single_score', True, bilinear, num_layers, inner_dim)

class HumanPairClassifier(nn.Module):
    def __init__(self, in_dim, model_type, bilinear=False, inner_layers = -1, inner_dim = 32):
        super(HumanPairClassifier, self).__init__()
        self.model_type = model_type
        self.bilinear = bilinear

        if bilinear:
            in_dim = in_dim * 2 + 2 * 2 * in_dim * in_dim
        else:
            in_dim = 2 * in_dim

        if model_type == 'binary':
            out_dim = 2
        elif model_type == 'ternary':
            out_dim = 3

        if inner_layers < 0:
            self.model = nn.Sequential(nn.Linear(in_dim, out_dim))
        else:
            model_list = [nn.Linear(in_dim, inner_dim*(2**inner_layers)), nn.ReLU()]
            for i in range(inner_layers):
                model_list.append(nn.Linear(inner_dim*(2**(inner_layers-i)), inner_dim * (2 ** (inner_layers - (i+1)))))
                model_list.append(nn.ReLU())
            model_list.append(nn.Linear(inner_dim, out_dim))
            self.model = nn.Sequential(*model_list)


    def forward(self, X):
        if self.bilinear:
            Y = X.unsqueeze(1) * X.unsqueeze(2)
            return self.model(torch.cat([X,Y.view(Y.size(0), -1)],1))
        else:
            return self.model(X)

class SingleScoreHumanClassifier(nn.Module):
    def __init__(self, in_dim, model_type, const_threshold = False, bilinear = False, inner_layers = -1, inner_dim=32):
        super(SingleScoreHumanClassifier, self).__init__()
        self.model_type = model_type
        self.bilinear = bilinear

        if bilinear:
            in_dim = in_dim * 2 + (2*in_dim)**2
        else:
            in_dim = in_dim * 2


        if inner_layers < 0:
            self.single_score = nn.Linear(in_dim, 1)
        else:
            model_list = [nn.Linear(in_dim, inner_dim*(2**inner_layers)), nn.ReLU()]
            for i in range(inner_layers):
                model_list.append(nn.Linear(inner_dim*(2**(inner_layers-i)), inner_dim * (2 ** (inner_layers - (i+1)))))
                model_list.append(nn.ReLU())
            model_list.append(nn.Linear(inner_dim, 1))
            self.single_score = nn.Sequential(*model_list)

        if const_threshold:
            self.register_buffer('threshold', torch.Tensor([0.0]))
        else:
            self.register_parameter('threshold', nn.Parameter(torch.randn(1)))
        self.register_parameter('scale', nn.Parameter(torch.randn(1)))

    def forward(self, X):
        if self.bilinear:
            Y = X.unsqueeze(1) * X.unsqueeze(2)
            score = self.single_score(torch.cat([X,Y.view(Y.size(0),-1)],1))
        else:
            score = self.single_score(X)
        thresh = torch.clamp(nn.functional.elu(self.threshold) + 1.0, min=0)
        scale = torch.clamp(nn.functional.elu(self.scale) + 1.0, min=1e-8)
        v1 = torch.sigmoid(scale * (score - thresh))
        v2 = 1.0 - torch.sigmoid(scale * (score + thresh))
        v3 = 1.0 - (v1 + v2)
        return torch.cat([v3,v1,v2],1)

class JointLoss(nn.Module):
    def __init__(self):
        super(JointLoss, self).__init__()
        self.w_ss_local = 2.0
        self.w_SAW = 1.0
        self.w_rs_local = 1.0
        self.w_reconstr = 2.0
        self.w_reconstr_real = 2.0
        self.w_rs_dense = 2.0
        self.w_ls = 2.0
        self.w_ss_dense = 4.0
        self.w_sp = 0.25
        self.w_IIW = 4.0
        self.w_feature = 0.75
        self.w_grad = 0.25
        self.w_human_similarity = 1e-3
        self.local_s_w = np.array([[0.5,    0.5,   0.5,    0.5,    0.5], \
                                   [0.5,    1 ,    1 ,     1,      0.5],\
                                   [0.5,    1,     1,      1,      0.5],\
                                   [0.5,    1,     1,      1,      0.5],\
                                   [0.5,    0.5,   0.5,    0.5,    0.5]])
        x = np.arange(-1, 2)
        y = np.arange(-1, 2)
        self.X, self.Y = np.meshgrid(x, y)
        # self.h_offset = [0,0,0,1,1,2,2,2,1]
        # self.w_offset = [0,1,2,0,2,0,1,2,1]
        self.total_loss = None
        self.running_stage = 0

        self.HumanPairClassifier = None
        self.pyr_levels = 1
        self.opt = None

    def BilateralRefSmoothnessLoss(self, pred_R, targets, att, num_features):
        # pred_R = pred_R.cpu()
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        N = pred_R.size(2) * pred_R.size(3)
        Z = (pred_R.size(1) * N )

        # grad_input = torch.FloatTensor(pred_R.size())
        # grad_input = grad_input.zero_()

        for i in range(pred_R.size(0)): # for each image
            B_mat = targets[att+'B_list'][i] # still list of blur sparse matrices
            S_mat = Variable(targets[att + 'S'][i].cuda(), requires_grad = False) # Splat and Slicing matrix
            n_vec = Variable(targets[att + 'N'][i].cuda(), requires_grad = False) # bi-stochatistic vector, which is diagonal matrix

            p = pred_R[i,:,:,:].view(pred_R.size(1),-1).t() # NX3
            # p'p
            # p_norm = torch.mm(p.t(), p)
            # p_norm_sum = torch.trace(p_norm)
            p_norm_sum = torch.sum(torch.mul(p,p))

            # S * N * p
            Snp = torch.mul(n_vec.repeat(1,pred_R.size(1)), p)
            sp_mm = Sparse()
            Snp = sp_mm(Snp, S_mat)

            Snp_1 = Snp.clone()
            Snp_2 = Snp.clone()

            # # blur
            for f in range(num_features+1):
                B_var1 = Variable(B_mat[f].cuda(), requires_grad = False)
                sp_mm1 = Sparse()
                Snp_1 = sp_mm1(Snp_1, B_var1)

                B_var2 = Variable(B_mat[num_features-f].cuda(), requires_grad = False)
                sp_mm2 = Sparse()
                Snp_2 = sp_mm2(Snp_2, B_var2)

            Snp_12 = Snp_1 + Snp_2
            pAp = torch.sum(torch.mul(Snp, Snp_12))


            total_loss = total_loss + ((p_norm_sum - pAp)/Z)

        total_loss = total_loss/pred_R.size(0)
        # average over all images
        return total_loss

    def SUNCGReconstLoss(self, R, S, mask, targets):
        rgb_img = Variable(targets['rgb_img'].cuda(), requires_grad = False)
        S = S.repeat(1,3,1,1)

        chromaticity = Variable(targets['chromaticity'].cuda(), requires_grad = False)
        R = torch.mul(chromaticity, R.repeat(1,3,1,1))

        return torch.mean( torch.pow(torch.mul(mask, rgb_img - torch.mul(R, S)), 2) )

    def IIWReconstLoss(self, R, S, targets):
        S = S.repeat(1,3,1,1)
        rgb_img = Variable(targets['rgb_img'].cuda(), requires_grad = False)

        # 1 channel
        chromaticity = Variable(targets['chromaticity'].cuda(), requires_grad = False)
        p_R = torch.mul(chromaticity, R.repeat(1,3,1,1))

        # return torch.mean( torch.mul(L, torch.pow( torch.log(rgb_img) - torch.log(p_R) - torch.log(S), 2)))
        return torch.mean( torch.pow( rgb_img - torch.mul(p_R, S), 2))


    def Ranking_Loss(self, prediction_R, judgements, is_flip):
        #ranking loss for each prediction feature
        tau = 0.25 #abs(I1 - I2)) ) #1.2 * (1 + math.fabs(math.log(I1) - math.log(I2) ) )

        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)

        num_valid_comparisons = 0

        num_valid_comparisons_ineq =0
        num_valid_comparisons_eq = 0

        total_loss_eq = Variable(torch.cuda.FloatTensor(1))
        total_loss_eq[0] = 0
        total_loss_ineq = Variable(torch.cuda.FloatTensor(1))
        total_loss_ineq[0] = 0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            #  remove unconfident point
            weight = c['darker_score']
            if weight < 0.5 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]

            if not point1['opaque'] or not point2['opaque']:
                continue

            # if is_flip:
                # l1 = prediction_R[:, int(point1['y'] * rows), cols - 1 - int( point1['x'] * cols)]
                # l2 = prediction_R[:, int(point2['y'] * rows), cols - 1 - int( point2['x'] * cols)]
            # else:
            l1 = prediction_R[:, int(point1['y'] * rows), int(point1['x'] * cols)]
            l2 = prediction_R[:, int(point2['y'] * rows), int(point2['x'] * cols)]

            l1_m = l1 #torch.mean(l1)
            l2_m = l2 #torch.mean(l2)

            # print(int(point1['y'] * rows), int(point1['x'] * cols), int(point2['y'] * rows), int(point2['x'] * cols), darker)
            # print(point1['y'], point1['x'], point2['y'], point2['x'], c['point1'], c['point2'])
            # print("===============================================================")
            # l2 > l1, l2 is brighter
            # if darker == '1' and ((l1_m.data[0] / l2_m.data[0]) > 1.0/tau):
            #     # loss =0
            #     loss =  weight * torch.mean((tau -  (l2_m / l1_m)))
            #     num_valid_comparisons += 1
            # # l1 > l2, l1 is brighter
            # elif darker == '2' and ((l2_m.data[0] / l1_m.data[0]) > 1.0/tau):
            #     # loss =0
            #     loss =  weight * torch.mean((tau -  (l1_m / l2_m)))
            #     num_valid_comparisons += 1
            # # is equal
            # elif darker == 'E':
            #     loss =  weight * torch.mean(torch.abs(l2 - l1))
            #     num_valid_comparisons += 1
            # else:
            #     loss = 0.0

            # l2 is brighter
            if darker == '1' and ((l1_m.data[0] - l2_m.data[0]) > - tau):
                # print("dark 1", l1_m.data[0] - l2_m.data[0])
                total_loss_ineq +=  weight * torch.mean( torch.pow( tau -  (l2_m - l1_m), 2)   )
                num_valid_comparisons_ineq += 1.
                # print("darker 1 loss", l2_m.data[0], l1_m.data[0], loss.data[0])
            # l1 > l2, l1 is brighter
            elif darker == '2' and ((l2_m.data[0] - l1_m.data[0]) > - tau):
                # print("dark 2", l2_m.data[0] - l1_m.data[0])
                total_loss_ineq += weight * torch.mean( torch.pow( tau -  (l1_m - l2_m),2)   )
                num_valid_comparisons_ineq += 1.
                # print("darker 2 loss", l2_m.data[0], l1_m.data[0], loss.data[0])
            elif darker == 'E':
                total_loss_eq +=  weight * torch.mean( torch.pow(l2 - l1,2) )
                num_valid_comparisons_eq += 1.
            else:
                loss = 0.0

        total_loss = total_loss_ineq + total_loss_eq
        num_valid_comparisons = num_valid_comparisons_eq + num_valid_comparisons_ineq

        # print("average eq loss", total_loss_eq.data[0]/(num_valid_comparisons_eq + 1e-6))
        # print("average ineq loss", total_loss_ineq.data[0]/(num_valid_comparisons_ineq + 1e-6))

        return total_loss/(num_valid_comparisons + 1e-6)

    def BatchRankingHingeLoss(self, prediction_R, judgements_eq, judgements_ineq, random_flip):
        eq_loss, ineq_loss = 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0
        tau = 0.425
        eps_ = .1

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)
        num_channel = prediction_R.size(0)

        # evaluate equality annotations densely
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()
            R_vec = prediction_R.view(num_channel, -1)
            # R_vec = torch.exp(R_vec)
            # I_vec = I.view(1, -1)

            y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            y_2 = torch.floor(judgements_eq[:,2] * rows).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_eq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = torch.floor(judgements_eq[:,3] * cols).long()

            # compute linear index for point 1
            # y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_eq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # compute linear index for point 2
            # y_2 = torch.floor(judgements_eq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_eq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec, 1, Variable(point_1_idx_linear, requires_grad = False))
            points_2_vec = torch.index_select(R_vec, 1, Variable(point_2_idx_linear, requires_grad = False))

            # I1_vec = torch.index_select(I_vec, 1, point_1_idx_linear)
            # I2_vec = torch.index_select(I_vec, 1, point_2_idx_linear)

            weight = Variable(judgements_eq[:,4], requires_grad = False)
            # weight = confidence#* torch.exp(4.0 * torch.abs(I1_vec - I2_vec) )

            # compute loss
            # eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.abs(points_1_vec - points_2_vec),0) ))

            eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.pow(points_1_vec - points_2_vec,2),0) ))
            num_valid_eq += judgements_eq.size(0)

        # compute inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            R_intensity = torch.mean(prediction_R, 0)
            # R_intensity = torch.log(R_intensity)
            R_vec_mean = R_intensity.view(1, -1)

            y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_ineq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            # y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec_mean, 1, Variable(point_1_idx_linear, requires_grad = False)).squeeze(0)
            points_2_vec = torch.index_select(R_vec_mean, 1, Variable(point_2_idx_linear, requires_grad = False)).squeeze(0)
            weight = Variable(judgements_ineq[:,4], requires_grad = False)

            # point 2 should be always darker than (<) point 1
            # compute loss
            relu_layer = nn.ReLU(True)
            # ineq_loss = torch.sum(torch.mul(weight, relu_layer(points_2_vec - points_1_vec + tau) ) )
            ineq_loss = torch.sum(torch.mul(weight, torch.pow( relu_layer(points_2_vec - points_1_vec + tau),2)  ) )
            # ineq_loss = torch.sum(torch.mul(weight, torch.pow(relu_layer(tau - points_1_vec/points_2_vec),2)))

            num_included = torch.sum( torch.ge(points_2_vec.data - points_1_vec.data, -tau).float().cuda() )
            # num_included = torch.sum(torch.ge(points_2_vec.data/points_1_vec.data, 1./tau).float().cuda())

            num_valid_ineq += num_included

        # avoid divide by zero
        return eq_loss/(num_valid_eq + 1e-8) +  ineq_loss/(num_valid_ineq + 1e-8)

    def BatchHumanBinaryClassifierLoss(self, prediction_R, judgements_eq, judgements_ineq, random_filp, human_pair_classifier):
        # human_pair_classifier should return two binary classifiers.  0th for "is equal" and 1st for "left or right"
        #ce_loss = nn.CrossEntropyLoss(reduction='none')
        #ce_loss.cuda()

        bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        bce_loss.cuda()

        eq_loss, ineq_eq_loss, ineq_pr_loss = 0, 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0

        # Collect equality annotations
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()
            feat_1, feat_2, weight, gt = self.getPyrValues(prediction_R, judgements_eq, random_filp)
            #gt_eq = gt * 0

            #eq_loss = torch.sum(weight * bce_loss(human_pair_classifier(torch.cat([feat_1, feat_2], 1)), gt_eq.long()))
            #num_valid_eq = torch.sum(weight)

            human_label = human_pair_classifier(torch.cat([feat_1, feat_2], 1))
            hl = human_label[:,0]
            gt_eq = torch.ones(hl.shape).cuda()

            eq_loss = torch.sum(weight * bce_loss(hl, gt_eq))
            num_valid_eq = torch.sum(weight)

        # collect inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            feat_1, feat_2, weight, gt = self.getPyrValues(prediction_R, judgements_ineq, random_filp)

            ineq_pred = human_pair_classifier(torch.cat([feat_1, feat_2],1))
            gt_ineq_eq = torch.zeros(ineq_pred.shape[0]).cuda()
            gt_ineq_pr = gt

            ineq_eq_loss = torch.sum(weight * bce_loss(ineq_pred[:,0], gt_ineq_eq))
            ineq_pr_loss = torch.sum(weight * bce_loss(ineq_pred[:,1], gt_ineq_pr))

            num_valid_ineq = torch.sum(weight)

        return .5 * ((eq_loss + ineq_eq_loss) / (num_valid_eq + num_valid_ineq + 1e-8) + ineq_pr_loss / (num_valid_ineq + 1e-8))

    def getPyrValues(self, prediction_R, judgements, random_flip):
        if self.pyr_levels > 1:
            gpyr = GaussianPyramid(prediction_R.size(0), self.pyr_levels)
            gpyr.cuda()
            pyr_pred_R = gpyr(prediction_R.unsqueeze(0))
        else:
            pyr_pred_R = [prediction_R.unsqueeze(0)]

        sel = torch.rand(judgements[:,1].shape).ge(.5).cuda()
        points_1_vec = []
        points_2_vec = []

        for pred_R in pyr_pred_R:
            pred_R = pred_R[0]
            rows = pred_R.size(1)
            cols = pred_R.size(2)
            num_channel = pred_R.size(0)

            R_vec = pred_R.contiguous().view(num_channel, -1)

            y_1 = torch.floor(judgements[:,0] * rows).long()
            y_2 = torch.floor(judgements[:,2] * rows).long()

            if random_flip:
                x_1 = cols - 1 - torch.floor(judgements[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements[:,1] * cols).long()
                x_2 = torch.floor(judgements[:,3] * cols).long()

            point_1_idx_linear = y_1 * cols + x_1
            point_2_idx_linear = y_2 * cols + x_2

            p1 = torch.cat([torch.masked_select(point_1_idx_linear, sel), torch.masked_select(point_2_idx_linear, 1-sel)],0)
            p2 = torch.cat([torch.masked_select(point_2_idx_linear, sel), torch.masked_select(point_1_idx_linear, 1-sel)],0)

            # extract all pairs of comparisions
            points_1_vec.append(torch.index_select(R_vec, 1, Variable(p1, requires_grad = False)))
            points_2_vec.append(torch.index_select(R_vec, 1, Variable(p2, requires_grad = False)))

        weight = Variable(judgements[:,4], requires_grad = False)
        gt_swapped = torch.cat([torch.zeros(torch.sum(sel).item()), torch.ones(torch.sum(1-sel).item())]).cuda()
        feat_1 = torch.cat(points_1_vec,0).transpose(0,1)
        feat_2 = torch.cat(points_2_vec,0).transpose(0,1)
        return feat_1, feat_2, weight, gt_swapped

    def BatchHumanClassifierLoss(self, prediction_R, judgements_eq, judgements_ineq, random_filp, human_pair_classifier):
        if human_pair_classifier.model_type == 'single_score':
            ce_loss = nn.NLLLoss(reduction='none')
            ce_loss.cuda()
        else:
            ce_loss = nn.CrossEntropyLoss(reduction='none')
            ce_loss.cuda()

        eq_loss, ineq_loss = 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0

        #rows = prediction_R.size(1)
        #cols = prediction_R.size(2)
        #num_channel = prediction_R.size(0)

        #R_vec = prediction_R.view(num_channel, -1)

        points_1_vec = []
        points_2_vec = []
        gts = []
        weights = []

        pred_hist = None
        gt_hist = None

        # Collect equality annotations
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()
            feat_1, feat_2, weight, gt = self.getPyrValues(prediction_R, judgements_eq, random_filp)
            gt_eq = gt * 0

            pred_eq = human_pair_classifier(torch.cat([feat_1, feat_2],1))

            eq_loss = torch.sum(weight * ce_loss(pred_eq, gt_eq.long()))
            num_valid_eq = torch.sum(weight)
            pred_hist = torch.histc(pred_eq.argmax(1).cpu().detach().float(), bins=3, min=0, max=2)
            gt_hist = torch.histc(gt_eq.cpu().detach().float(), bins=3, min=0, max=2)

        # collect inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            feat_1, feat_2, weight, gt = self.getPyrValues(prediction_R, judgements_ineq, random_filp)
            gt_ineq = gt + 1

            pred_ineq = human_pair_classifier(torch.cat([feat_1, feat_2], 1))

            ineq_loss = torch.sum(weight * ce_loss(pred_ineq, gt_ineq.long()))
            num_valid_ineq = torch.sum(weight)
            ineq_hist = torch.histc(pred_ineq.argmax(1).cpu().detach().float(), bins=3, min=0, max=2)
            gt_ineq_hist = torch.histc(gt_ineq.cpu().detach().float(), bins=3, min=0, max=2)
            if pred_hist is not None:
                pred_hist = pred_hist + ineq_hist
                gt_hist = gt_hist + gt_ineq_hist
            else:
                pred_hist = ineq_hist
                gt_hist = gt_ineq_hist

        print pred_hist, gt_hist
        # avoid divide by zero
        #print 'eq, ineq', eq_loss/(num_valid_eq + 1e-8), ineq_loss/(num_valid_ineq + 1e-8)
        # equal weight to eq, >, < for image
        return eq_loss/(num_valid_eq + 1e-8) +  2 * ineq_loss/(num_valid_ineq + 1e-8)
        # no weighting
        #return (eq_loss + ineq_loss) / (num_valid_eq + num_valid_ineq + 1e-8)
        # equal weight to eq, ineq for image
        #return eq_loss/(num_valid_eq + 1e-8) +  ineq_loss/(num_valid_ineq + 1e-8)

    def BatchRankingLoss(self, prediction_R, judgements_eq, judgements_ineq, random_filp):
        eq_loss, ineq_loss = 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0
        tau = 0.425
        assert(prediction_R.size(0) == 1)

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)
        num_channel = prediction_R.size(0)

        # evaluate equality annotations densely
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()
            R_vec = prediction_R.view(num_channel, -1)
            # R_vec = torch.exp(R_vec)
            # I_vec = I.view(1, -1)

            y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            y_2 = torch.floor(judgements_eq[:,2] * rows).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_eq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_eq[:,1] * cols).long()
                x_2 = torch.floor(judgements_eq[:,3] * cols).long()

            # compute linear index for point 1
            # y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_eq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # compute linear index for point 2
            # y_2 = torch.floor(judgements_eq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_eq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec, 1, Variable(point_1_idx_linear, requires_grad = False))
            points_2_vec = torch.index_select(R_vec, 1, Variable(point_2_idx_linear, requires_grad = False))

            # I1_vec = torch.index_select(I_vec, 1, point_1_idx_linear)
            # I2_vec = torch.index_select(I_vec, 1, point_2_idx_linear)

            weight = Variable(judgements_eq[:,4], requires_grad = False)
            # weight = confidence#* torch.exp(4.0 * torch.abs(I1_vec - I2_vec) )

            # compute loss
            # eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.abs(points_1_vec - points_2_vec),0) ))
            eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.pow(points_1_vec - points_2_vec,2),0) ))
            num_valid_eq += torch.sum(weight)

        # compute inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            R_intensity = torch.mean(prediction_R, 0)
            # R_intensity = torch.log(R_intensity)
            R_vec_mean = R_intensity.view(1, -1)

            y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            if random_filp:
                x_1 = cols - 1 - torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = cols - 1 - torch.floor(judgements_ineq[:,3] * cols).long()
            else:
                x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
                x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            # y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec_mean, 1, Variable(point_1_idx_linear, requires_grad = False)).squeeze(0)
            points_2_vec = torch.index_select(R_vec_mean, 1, Variable(point_2_idx_linear, requires_grad = False)).squeeze(0)
            weight = Variable(judgements_ineq[:,4], requires_grad = False)

            # point 2 should be always darker than (<) point 1
            # compute loss
            relu_layer = nn.ReLU(True)
            # ineq_loss = torch.sum(torch.mul(weight, relu_layer(points_2_vec - points_1_vec + tau) ) )
            ineq_loss = torch.sum(torch.mul(weight, torch.pow( relu_layer(points_2_vec - points_1_vec + tau),2)  ) )
            # ineq_loss = torch.sum(torch.mul(weight, torch.pow(relu_layer(tau - points_1_vec/points_2_vec),2)))

            #num_included = torch.sum( torch.ge(points_2_vec.data - points_1_vec.data, -tau).float().cuda() )
            # num_included = torch.sum(torch.ge(points_2_vec.data/points_1_vec.data, 1./tau).float().cuda())

            num_valid_ineq += torch.sum(weight)
            #num_valid_ineq += num_included

        # avoid divide by zero
        return eq_loss/(num_valid_eq + 1e-8) +  ineq_loss/(num_valid_ineq + 1e-8)

    def ShadingPenaltyLoss(self, S):
        return torch.mean(torch.pow(S - 0.5,2) )
        # return torch.sum( torch.mul(sky_mask, torch.abs(S - np.log(0.5))/num_val_pixels ))

    def AngleLoss(self, prediction_n, targets):
        mask = Variable(targets['mask'].cuda(), requires_grad = False)
        normal = Variable(targets['normal'].cuda(), requires_grad = False)
        num_valid = torch.sum(mask[:,0,:,:])
        # compute dot product
        angle_loss = - torch.sum( torch.mul(mask, torch.mul(prediction_n, normal)), 1)
        return 1 + torch.sum(angle_loss)/num_valid


    def GradientLoss(self, prediction_n, mask, gt_n):
        N = torch.sum(mask)

        # horizontal angle difference
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = prediction_n[:,:,:,0:-2] - prediction_n[:,:,:,2:]
        h_gradient_gt = gt_n[:,:,:,0:-2] -  gt_n[:,:,:,2:]
        h_gradient_loss = torch.mul(h_mask, torch.abs(h_gradient - h_gradient_gt))


        # Vertical angle difference
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = prediction_n[:,:,0:-2,:] -  prediction_n[:,:,2:,:]
        v_gradient_gt = gt_n[:,:,0:-2,:] - gt_n[:,:,2:,:]
        v_gradient_loss = torch.mul(v_mask, torch.abs(v_gradient - v_gradient_gt))


        gradient_loss = torch.sum(h_gradient_loss) + torch.sum(v_gradient_loss)
        gradient_loss = gradient_loss/(N*2.0)

        return gradient_loss

    def SmoothLoss(self, prediction_n, mask):
        N = torch.sum(mask[:,0,:,:])

        # horizontal angle difference
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.sum( torch.mul(h_mask, torch.mul(prediction_n[:,:,:,0:-2], prediction_n[:,:,:,2:])), 1)
        h_gradient_loss = 1 - torch.sum(h_gradient)/N

        # Vertical angle difference
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.sum( torch.mul(v_mask, torch.mul(prediction_n[:,:,0:-2,:], prediction_n[:,:,2:,:])), 1)
        v_gradient_loss = 1 - torch.sum(v_gradient)/N

        gradient_loss = h_gradient_loss + v_gradient_loss

        return gradient_loss

    def UncertaintyLoss(self, prediction_n, uncertainty, targets):
        uncertainty = torch.squeeze(uncertainty, 1)

        mask = Variable(targets['mask'].cuda(), requires_grad = False)
        normal = Variable(targets['normal'].cuda(), requires_grad = False)
        num_valid = torch.sum(mask[:,0,:,:])

        angle_diff = ( torch.sum( torch.mul(prediction_n, normal), 1) + 1.0) * 0.5
        uncertainty_loss = torch.sum( torch.mul(mask[:,0,:,:], torch.pow(uncertainty - angle_diff, 2) ) )
        return uncertainty_loss/num_valid

    def MaskLocalSmoothenessLoss(self, R, M, targets):
        h = R.size(2)
        w = R.size(3)
        num_c = R.size(1)

        half_window_size = 1
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        mask_center = M[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
                        half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size]]

        R_center = R[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
                         half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size] ]

        c_idx = 0

        for k in range(0,half_window_size*2+1):
            for l in range(0,half_window_size*2+1):
                # albedo_weights = Variable(targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,num_c,1,1).float().cuda(), requires_grad = False)
                R_N = R[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l],
                            half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
                mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l],
                                half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]

                composed_M = torch.mul(mask_N, mask_center)

                # albedo_weights = torch.mul(albedo_weights, composed_M)

                r_diff = torch.mul( composed_M, torch.pow(R_center - R_N,2)  )
                total_loss  = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1


        return total_loss/(8.0 * num_c)


    def LocalAlebdoSmoothenessLoss(self, R, targets, scale_idx):
        h = R.size(2)
        w = R.size(3)
        num_c = R.size(1)

        half_window_size = 1
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0

        R_center = R[:,:,half_window_size + self.Y[half_window_size,half_window_size]:h-half_window_size + self.Y[half_window_size,half_window_size], \
                         half_window_size + self.X[half_window_size,half_window_size]:w-half_window_size + self.X[half_window_size,half_window_size] ]

        c_idx = 0

        for k in range(0,half_window_size*2+1):
            for l in range(0,half_window_size*2+1):
                albedo_weights = targets["r_w_s"+str(scale_idx)][:,c_idx,:,:].unsqueeze(1).repeat(1,num_c,1,1).float().cuda()
                R_N = R[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
                # mask_N = M[:,:,half_window_size + self.Y[k,l]:h- half_window_size + self.Y[k,l], half_window_size + self.X[k,l]: w-half_window_size + self.X[k,l] ]
                # composed_M = torch.mul(mask_N, mask_center)
                # albedo_weights = torch.mul(albedo_weights, composed_M)
                r_diff = torch.mul( Variable(albedo_weights, requires_grad = False), torch.abs(R_center - R_N)  )

                total_loss  = total_loss + torch.mean(r_diff)
                c_idx = c_idx + 1


        return total_loss/(8.0 * num_c)


    def Data_Loss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)
        s1 = torch.sum( torch.pow(log_diff,2) )/N
        s2 = torch.pow(torch.sum(log_diff),2)/(N*N)
        data_loss = s1 - s2
        return data_loss

    def L2GradientMatchingLoss(self, log_prediction, mask, log_gt):
        N = torch.sum(mask)
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)

        v_gradient = torch.pow(log_diff[:,:,0:-2,:] - log_diff[:,:,2:,:],2)
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.pow(log_diff[:,:,:,0:-2] - log_diff[:,:,:,2:],2)
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))
        gradient_loss = gradient_loss/N

        return gradient_loss

    def L1GradientMatchingLoss(self, log_prediction, mask, log_gt):
        N = torch.sum( mask )
        log_diff = log_prediction - log_gt
        log_diff = torch.mul(log_diff, mask)

        v_gradient = torch.abs(log_diff[:,:,0:-2,:] - log_diff[:,:,2:,:])
        v_mask = torch.mul(mask[:,:,0:-2,:], mask[:,:,2:,:])
        v_gradient = torch.mul(v_gradient, v_mask)

        h_gradient = torch.abs(log_diff[:,:,:,0:-2] - log_diff[:,:,:,2:])
        h_mask = torch.mul(mask[:,:,:,0:-2], mask[:,:,:,2:])
        h_gradient = torch.mul(h_gradient, h_mask)

        gradient_loss = (torch.sum(h_gradient) + torch.sum(v_gradient))/2.0
        gradient_loss = gradient_loss/N

        return gradient_loss

    def L1Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum( mask )
        diff = torch.mul(mask, torch.abs(prediction_n - gt))
        return torch.sum(diff)/num_valid

    def L2Loss(self, prediction_n, mask, gt):
        num_valid = torch.sum( mask )

        diff = torch.mul(mask, torch.pow(prediction_n - gt,2))
        return torch.sum(diff)/(num_valid + 1e-8)

    def HuberLoss(self, prediction, mask, gt):
        tau = 1.0
        num_valid = torch.sum(mask)

        diff_L1 = torch.abs(prediction - gt)
        diff_L2 = torch.pow(prediction - gt ,2)

        mask_L2 = torch.le(diff_L1, tau).float().cuda()
        mask_L1 = 1.0 - mask_L2

        L2_loss = 0.5 * torch.sum(torch.mul(mask, torch.mul(mask_L2, diff_L2)))
        L1_loss = torch.sum(torch.mul(mask, torch.mul(mask_L1, diff_L1))) - 0.5

        final_loss = (L2_loss + L1_loss)/num_valid
        return final_loss

    def CCLoss(self, prediction_S, saw_mask, gts, num_cc):
        diff = prediction_S - gts
        total_loss = Variable(torch.cuda.FloatTensor(1))
        total_loss[0] = 0
        num_regions = 0

        # for each prediction
        for i in range(prediction_S.size(0)):
            log_diff = diff[i,:,:,:]
            mask = saw_mask[i,:,:,:].int()

            for k in range(1, num_cc[i]+1):
                new_mask = (mask == k).float().cuda()

                masked_log_diff = torch.mul(new_mask, log_diff)
                N = torch.sum(new_mask)

                s1 = torch.sum( torch.pow(masked_log_diff,2) )/N
                s2 = torch.pow(torch.sum(masked_log_diff),2)/(N*N)
                total_loss += (s1 - s2)
                num_regions +=1

        return total_loss/(num_regions + 1e-6)


    def SAWLoss(self, prediction_S, targets):
        # Shading smoothness ignore mask region
        lambda_1, lambda_2 = 0.1, 1.

        # saw_mask_0 = Variable(targets['saw_mask_0'].cuda(), requires_grad = False)
        # prediction_S_1 = prediction_S[:,:,::2,::2]
        # prediction_S_2 = prediction_S_1[:,:,::2,::2]
        # prediction_S_3 = prediction_S_2[:,:,::2,::2]

        # mask_0 = saw_mask_0
        # mask_1 = mask_0[:,:,::2,::2]
        # mask_2 = mask_1[:,:,::2,::2]
        # mask_3 = mask_2[:,:,::2,::2]

        # saw_loss_0 = self.w_ss_local * self.MaskLocalSmoothenessLoss(prediction_S, mask_0, targets)
        # saw_loss_0 += self.w_ss_local * 0.5 * self.MaskLocalSmoothenessLoss(prediction_S_1, mask_1, targets)
        # saw_loss_0 += self.w_ss_local * 0.333 * self.MaskLocalSmoothenessLoss(prediction_S_2, mask_2, targets)
        # saw_loss_0 += self.w_ss_local * 0.25 * self.MaskLocalSmoothenessLoss(prediction_S_3, mask_3, targets)

        # shadow boundary
        saw_mask_1 = Variable(targets['saw_mask_1'].cuda(), requires_grad = False)
        linear_I = torch.mean( Variable(targets['rgb_img'].cuda(), requires_grad = False),1)
        linear_I = linear_I.unsqueeze(1)
        linear_I[linear_I < 1e-4] = 1e-4

        # linear_I = linear_I.data[0,0,:,:].cpu().numpy()
        # srgb_img = np.transpose(linear_I, (1 , 2 ,0))
        # mask_1 = saw_mask_1.data[0,0,:,:].cpu().numpy()
        # R_np = np.transpose(R_np, (1 , 2 ,0

        # print(targets['num_mask_1'][0])
        # plt.figure()
        # plt.imshow(mask_1, cmap='gray')
        # plt.show()  # display i
        # plt.figure()
        # plt.imshow(linear_I, cmap='gray')
        # plt.show()  # display i
        # sys.exit()

        saw_loss_1 = lambda_1 * self.CCLoss(prediction_S, saw_mask_1, torch.log(linear_I), targets['num_mask_1'])
        # smooth region
        saw_mask_2 = Variable(targets['saw_mask_2'].cuda(), requires_grad = False)
        saw_loss_2 = lambda_2 * self.CCLoss(prediction_S, saw_mask_2, 0, targets['num_mask_2'])

        # print("saw_loss_1 ", saw_loss_1.data[0])
        # print("saw_loss_2 ", saw_loss_2.data[0])

        return saw_loss_2 + saw_loss_1


    def DirectFramework(self, prediction, gt, mask):

        w_data = 1.0
        w_grad = 0.5
        final_loss = w_data * self.L2Loss(prediction, mask, gt)

        # level 0
        prediction_1 = prediction[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction , mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    # all parameter in log space, presumption
    def ScaleInvarianceFramework(self, prediction, gt, mask, w_grad):

        assert(prediction.size(1) == gt.size(1))
        assert(prediction.size(1) == mask.size(1))

        w_data = 1.0
        final_loss = w_data * self.Data_Loss(prediction, mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction , mask, gt)

        # level 0
        prediction_1 = prediction[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss

    def LinearScaleInvarianceFramework(self, prediction, gt, mask, w_grad):

        assert(prediction.size(1) == gt.size(1))
        assert(prediction.size(1) == mask.size(1))

        w_data = 1.0
        # w_grad = 0.5
        gt_vec = gt[mask > 0.1]
        pred_vec = prediction[mask > 0.1]
        gt_vec = gt_vec.unsqueeze(1).float().cpu()
        pred_vec = pred_vec.unsqueeze(1).float().cpu()

        if gt_vec.size(0) == 0:
            #return torch.tensor(0)
            return torch.tensor(0).type_as(prediction)

        scale, _ = torch.gels(gt_vec.data, pred_vec.data)
        scale = scale[0,0]

        # print("scale" , scale)
        # sys.exit()
        prediction_scaled = prediction * scale
        final_loss = w_data * self.L2Loss(prediction_scaled, mask, gt)

        prediction_1 = prediction_scaled[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_scaled , mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss


    def WeightedLinearScaleInvarianceFramework(self, prediction, gt, mask, w_grad):
        w_data = 1.0

        assert(prediction.size(1) == gt.size(1))
        assert(prediction.size(1) == mask.size(1))

        if torch.sum(mask.data) < 10:
            return 0

        # w_grad = 0.5
        gt_vec = gt[mask > 0.1]
        pred_vec = prediction[mask > 0.1]
        gt_vec = gt_vec.unsqueeze(1).float().cpu()
        pred_vec = pred_vec.unsqueeze(1).float().cpu()

        scale, _ = torch.gels(gt_vec.data, pred_vec.data)
        scale = scale[0,0]

        prediction_scaled = prediction * scale

        ones_matrix = Variable(torch.zeros(gt.size(0), gt.size(1), gt.size(2), gt.size(3)) + 1, requires_grad = False)
        weight = torch.min(1/gt,  ones_matrix.float().cuda())
        weight_mask = torch.mul(weight, mask)

        final_loss = w_data * self.L2Loss(prediction_scaled, weight_mask, gt)

        prediction_1 = prediction_scaled[:,:,::2,::2]
        prediction_2 = prediction_1[:,:,::2,::2]
        prediction_3 = prediction_2[:,:,::2,::2]

        mask_1 = weight_mask[:,:,::2,::2]
        mask_2 = mask_1[:,:,::2,::2]
        mask_3 = mask_2[:,:,::2,::2]

        gt_1 = gt[:,:,::2,::2]
        gt_2 = gt_1[:,:,::2,::2]
        gt_3 = gt_2[:,:,::2,::2]

        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_scaled , weight_mask, gt)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_1, mask_1, gt_1)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_2, mask_2, gt_2)
        final_loss += w_grad * self.L1GradientMatchingLoss(prediction_3, mask_3, gt_3)

        return final_loss


    def SUNCGBatchRankingLoss(self, prediction_R, judgements_eq, judgements_ineq):
        eq_loss, ineq_loss = 0, 0
        num_valid_eq = 0
        num_valid_ineq = 0
        tau = 0.4

        rows = prediction_R.size(1)
        cols = prediction_R.size(2)
        num_channel = prediction_R.size(0)

        # evaluate equality annotations densely
        if judgements_eq.size(1) > 2:
            judgements_eq = judgements_eq.cuda()

            R_vec = prediction_R.view(num_channel, -1)
            # R_vec = torch.exp(R_vec)

            y_1 = judgements_eq[:,0].long()
            y_2 = judgements_eq[:,2].long()

            # if random_filp:
                # x_1 = cols - 1 - judgements_eq[:,1].long()
                # x_2 = cols - 1 - judgements_eq[:,3].long()
            # else:
            x_1 = judgements_eq[:,1].long()
            x_2 = judgements_eq[:,3].long()

            # compute linear index for point 1
            # y_1 = torch.floor(judgements_eq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_eq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # compute linear index for point 2
            # y_2 = torch.floor(judgements_eq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_eq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec, 1, Variable(point_1_idx_linear, requires_grad = False))
            points_2_vec = torch.index_select(R_vec, 1, Variable(point_2_idx_linear, requires_grad = False))

            # I1_vec = torch.index_select(I_vec, 1, point_1_idx_linear)
            # I2_vec = torch.index_select(I_vec, 1, point_2_idx_linear)

            # weight = Variable(judgements_eq[:,4], requires_grad = False)
            # weight = confidence#* torch.exp(4.0 * torch.abs(I1_vec - I2_vec) )

            # compute Loss
            # eq_loss = torch.sum(torch.mul(weight, torch.mean(torch.abs(points_1_vec - points_2_vec),0) ))
            eq_loss = torch.sum( torch.mean( torch.pow(points_1_vec - points_2_vec,2) ,0) )
            num_valid_eq += judgements_eq.size(0)

        # # compute inequality annotations
        if judgements_ineq.size(1) > 2:
            judgements_ineq = judgements_ineq.cuda()
            R_intensity = torch.mean(prediction_R, 0)
            # R_intensity = torch.log(R_intensity)
            R_vec_mean = R_intensity.view(1, -1)

            y_1 = judgements_ineq[:,0].long()
            y_2 = judgements_ineq[:,2].long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()

            x_1 = judgements_ineq[:,1].long()
            x_2 = judgements_ineq[:,3].long()

            # y_1 = torch.floor(judgements_ineq[:,0] * rows).long()
            # x_1 = torch.floor(judgements_ineq[:,1] * cols).long()
            point_1_idx_linear = y_1 * cols + x_1
            # y_2 = torch.floor(judgements_ineq[:,2] * rows).long()
            # x_2 = torch.floor(judgements_ineq[:,3] * cols).long()
            point_2_idx_linear = y_2 * cols + x_2

            # extract all pairs of comparisions
            points_1_vec = torch.index_select(R_vec_mean, 1, Variable(point_1_idx_linear, requires_grad = False)).squeeze(0)
            points_2_vec = torch.index_select(R_vec_mean, 1, Variable(point_2_idx_linear, requires_grad = False)).squeeze(0)

            # point 2 should be always darker than (<) point 1
            # compute loss
            relu_layer = nn.ReLU(True)
            # ineq_loss = torch.sum(torch.mul(weight, relu_layer(points_2_vec - points_1_vec + tau) ) )
            ineq_loss = torch.sum(torch.pow( relu_layer(points_2_vec - points_1_vec + tau),2) )
            # ineq_loss = torch.sum(torch.mul(weight, torch.pow(relu_layer(tau - points_1_vec/points_2_vec),2)))
            num_included = torch.sum( torch.ge(points_2_vec.data - points_1_vec.data, -tau).float().cuda() )
            # num_included = torch.sum(torch.ge(points_2_vec.data/points_1_vec.data, 1./tau).float().cuda())

            num_valid_ineq += judgements_ineq.size(0)
            #num_val_inex += num_included

        # avoid divide by zero
        return (eq_loss)/(num_valid_eq + 1e-8) + ineq_loss/(num_valid_ineq + 1e-8)


    def IIW_loss(self, prediction, targets):
        num_images = prediction.size(0)
        total_iiw_loss = Variable(torch.cuda.FloatTensor(1))
        total_iiw_loss[0] = 0

        for i in range(0, num_images):
            # judgements = json.load(open(targets["judgements_path"][i]))
            # total_iiw_loss += self.w_IIW * self.Ranking_Loss(prediction_R[i,:,:,:], judgements, random_filp)
            if self.opt.use_base_IIW:
                judgements_eq = targets["gt_eq_mat"][i]
                judgements_ineq = targets["gt_ineq_mat"][i]
            else:
                judgements_eq = targets["eq_mat"][i]
                judgements_ineq = targets["ineq_mat"][i]
            random_filp = targets["random_filp"][i]
            if self.HumanPairClassifier is not None:
                if self.HumanPairClassifier.model_type == 'ternary':
                    total_iiw_loss += self.w_IIW * self.BatchHumanClassifierLoss(prediction[i,:,:,:], judgements_eq, judgements_ineq, random_filp, self.HumanPairClassifier)
                elif self.HumanPairClassifier.model_type == 'binary':
                    total_iiw_loss += self.w_IIW * self.BatchHumanBinaryClassifierLoss(prediction[i,:,:,:], judgements_eq, judgements_ineq, random_filp, self.HumanPairClassifier)
                elif self.HumanPairClassifier.model_type == 'single_score':
                    total_iiw_loss += self.w_IIW * self.BatchHumanClassifierLoss(prediction[i,:,:,:], judgements_eq, judgements_ineq, random_filp, self.HumanPairClassifier)

            else:
                total_iiw_loss += self.w_IIW * self.BatchRankingLoss(prediction[i,:,:,:], judgements_eq, judgements_ineq, random_filp)
        return (total_iiw_loss)/num_images

    def __call__(self, input_images, prediction_R, prediction_R_human, prediction_S, targets, data_set_name, epoch):

        lambda_CG = 0.5

        if data_set_name == "IIW":
            print("IIW Loss")
            if not self.opt.pretrained_cgi:
                num_images = prediction_R.size(0)
                # Albedo smoothness term
                # rs_loss =  self.w_rs_dense * self.BilateralRefSmoothnessLoss(prediction_R, targets, 'R', 5)
                # multi-scale smoothness term
                prediction_R_1 = prediction_R[:,:,::2,::2]
                prediction_R_2 = prediction_R_1[:,:,::2,::2]
                prediction_R_3 = prediction_R_2[:,:,::2,::2]

                rs_loss = self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R, targets,0)
                rs_loss = rs_loss +  0.5 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_1, targets,1)
                rs_loss = rs_loss +  0.3333 * self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R_2, targets,2)
                rs_loss = rs_loss +  0.25 * self.w_rs_local   * self.LocalAlebdoSmoothenessLoss(prediction_R_3, targets,3)

                prediction_R_human_1 = prediction_R_human[:,:,::2,::2]
                prediction_R_human_2 = prediction_R_human_1[:,:,::2,::2]
                prediction_R_human_3 = prediction_R_human_2[:,:,::2,::2]

                rs_human_loss = self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R_human, targets,0)
                rs_human_loss = rs_human_loss +  0.5 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_human_1, targets,1)
                rs_human_loss = rs_human_loss +  0.3333 * self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R_human_2, targets,2)
                rs_human_loss = rs_human_loss +  0.25 * self.w_rs_local   * self.LocalAlebdoSmoothenessLoss(prediction_R_human_3, targets,3)

                rs_loss = rs_loss * .5 + rs_human_loss + .5

                #similarity = self.w_human_similarity * (torch.abs(prediction_R.detach() - prediction_R_human)).mean()

                # # Lighting smoothness Loss
                ss_loss = self.w_ss_dense * self.BilateralRefSmoothnessLoss(prediction_S, targets, 'S', 2)
                # # Reconstruction Loss
                reconstr_loss = self.w_reconstr_real * self.IIWReconstLoss(torch.exp(prediction_R), \
                                                        torch.exp(prediction_S), targets)

            # IIW Loss
            if self.opt.detach_iiw_loss:
                detach_channels = 1
                if self.opt.append_chroma:
                    detach_channels += 3

                if prediction_R_human.size(1) == detach_channels:
                    total_iiw_loss = self.IIW_loss(prediction_R_human.detach(), targets)
                else:
                    prediction_R_human_detach = torch.cat([prediction_R_human[:,:detach_channels,...].detach(), prediction_R_human[:,detach_channels:,...]], 1)
                    total_iiw_loss = self.IIW_loss(prediction_R_human_detach, targets)
            else:
                total_iiw_loss = self.IIW_loss(prediction_R_human, targets)

            # print("reconstr_loss ", reconstr_loss.data[0])
            # print("rs_loss ", rs_loss.data[0])
            # print("ss_loss ", ss_loss.data[0])
            # print("total_iiw_loss ", total_iiw_loss.data[0])

            if self.opt.pretrained_cgi:
                total_loss = total_iiw_loss + 0 * prediction_R.sum() + 0 * prediction_S.sum() + 0 * prediction_R_human.sum()
            else:
                total_loss = total_iiw_loss + reconstr_loss + rs_loss + ss_loss

        elif data_set_name == "Render":
            print("Render LOSS")
            mask = Variable(targets['mask'].cuda(), requires_grad = False)
            mask_R = mask[:,0,:,:].unsqueeze(1).repeat(1,prediction_R.size(1),1,1)
            mask_S = mask[:,0,:,:].unsqueeze(1).repeat(1,prediction_S.size(1),1,1)
            mask_img = mask[:,0,:,:].unsqueeze(1).repeat(1,input_images.size(1),1,1)

            gt_R = Variable(targets['gt_R'].cuda(), requires_grad = False)
            gt_S = Variable(targets['gt_S'].cuda(), requires_grad = False)

            R_loss = lambda_CG *self.LinearScaleInvarianceFramework(torch.exp(prediction_R), gt_R, mask_R, 0.5)
            S_loss = lambda_CG * self.LinearScaleInvarianceFramework(torch.exp(prediction_S), gt_S, mask_S, 0.5)

            # using ScaleInvarianceFramework might achieve better performance if we train on both IIW and SAW,
            # but LinearScaleInvarianceFramework could produce better perforamnce if trained on CGIntrinsics only

            reconstr_loss = lambda_CG  * self.w_reconstr * self.SUNCGReconstLoss(torch.exp(prediction_R), torch.exp(prediction_S), mask_img, targets)


            total_loss = R_loss + S_loss + reconstr_loss

        elif data_set_name == "CGIntrinsics":
# ============================================================================================== This is scale invariance loss ===============
            print("CGIntrinsics LOSS")

            mask = Variable(targets['mask'].cuda(), requires_grad = False)
            mask_R = mask[:,0,:,:].unsqueeze(1).repeat(1,prediction_R.size(1),1,1)
            mask_S = mask[:,0,:,:].unsqueeze(1).repeat(1,prediction_S.size(1),1,1)
            mask_img = mask[:,0,:,:].unsqueeze(1).repeat(1,input_images.size(1),1,1)

            gt_R = Variable(targets['gt_R'].cuda(), requires_grad = False)
            gt_S = Variable(targets['gt_S'].cuda(), requires_grad = False)

            R_loss = lambda_CG *self.LinearScaleInvarianceFramework(torch.exp(prediction_R), gt_R, mask_R, 0.5)
            S_loss = lambda_CG * self.LinearScaleInvarianceFramework(torch.exp(prediction_S), gt_S, mask_S, 0.5)

            # using ScaleInvarianceFramework might achieve better performance if we train on both IIW and SAW,
            # but LinearScaleInvarianceFramework could produce better perforamnce if trained on CGIntrinsics only

            reconstr_loss = lambda_CG  * self.w_reconstr * self.SUNCGReconstLoss(torch.exp(prediction_R), torch.exp(prediction_S), mask_img, targets)

            # Why put this? Because some ground truth shadings are nosiy
            Ss_loss = lambda_CG * self.w_ss_dense *  self.BilateralRefSmoothnessLoss(prediction_S, targets, 'S', 2)

            total_iiw_loss = 0

            for i in range(0, prediction_R.size(0)):
                judgements_eq = targets["eq_mat"][i]
                judgements_ineq = targets["ineq_mat"][i]
                random_filp = targets["random_filp"][i]
                total_iiw_loss += lambda_CG * self.SUNCGBatchRankingLoss(prediction_R[i,:,:,:], judgements_eq, judgements_ineq)

            total_iiw_loss = total_iiw_loss/prediction_R.size(0)

            # print("R_loss ", R_loss.data[0])
            # print("S_loss ", S_loss.data[0])
            # print("reconstr_loss ", reconstr_loss.data[0])
            # print("Ss_loss ", Ss_loss.data[0])
            # print("SUNCGBatchRankingLoss   ", total_iiw_loss.data[0])

            total_loss = R_loss + S_loss + reconstr_loss + Ss_loss + total_iiw_loss

        elif data_set_name == "SAW":
            print("SAW Loss")

            prediction_R_1 = prediction_R[:,:,::2,::2]
            prediction_R_2 = prediction_R_1[:,:,::2,::2]
            prediction_R_3 = prediction_R_2[:,:,::2,::2]

            rs_loss = self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R, targets,0)
            rs_loss = rs_loss +  0.5 * self.w_rs_local * self.LocalAlebdoSmoothenessLoss(prediction_R_1, targets,1)
            rs_loss = rs_loss +  0.3333 * self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R_2, targets,2)
            rs_loss = rs_loss +  0.25 * self.w_rs_local  * self.LocalAlebdoSmoothenessLoss(prediction_R_3, targets,3)

            reconstr_loss = self.w_reconstr_real * self.IIWReconstLoss(torch.exp(prediction_R), \
                                                    torch.exp(prediction_S), targets)

            ss_loss = self.w_ss_dense * self.BilateralRefSmoothnessLoss(prediction_S, targets, 'S', 2)

            SAW_loss = self.w_SAW * self.SAWLoss(prediction_S, targets)

            # print("rs_loss ", rs_loss.data[0])
            # print("SAW_loss ", SAW_loss.data[0])
            # print("reconstr_loss ", reconstr_loss.data[0])
            # print("ss_loss ", ss_loss.data[0])

            total_loss = rs_loss + SAW_loss + reconstr_loss + ss_loss

        else:
            print("NORMAL Loss")
            sys.exit()

        self.total_loss = total_loss

        return total_loss.data.item()

    def compute_whdr_classifier(self, reflectance, judgements, human_classifier, eq_threshold = 0):
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = reflectance.shape[0:2]

        error_sum = 0.0
        error_equal_sum = 0.0
        error_inequal_sum = 0.0

        weight_sum = 0.0
        weight_equal_sum = 0.0
        weight_inequal_sum = 0.0

        # convert to pyramid
        if self.pyr_levels > 1:
            reflectance_tensor = torch.Tensor(np.transpose(reflectance, (2,0,1))).cuda()
            gauss_pyramid = GaussianPyramid(reflectance_tensor.size(0), self.pyr_levels)
            gauss_pyramid.cuda()
            gpyr = gauss_pyramid(reflectance_tensor.unsqueeze(0))
        else:
            reflectance_tensor = torch.Tensor(np.transpose(reflectance, (2,0,1))).cuda()
            gpyr = [reflectance_tensor.unsqueeze(0)]

        counts = [0,0,0]
        gt_counts = [0,0,0]
        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            p1 = []
            p2 = []
            for img in gpyr:
                rows, cols = img.shape[2:]
                p1.append(img[0, : ,int(point1['y'] * rows), int(point1['x'] * cols)].unsqueeze(0))
                p2.append(img[0, : ,int(point2['y'] * rows), int(point2['x'] * cols)].unsqueeze(0))
            p1 = torch.cat(p1,1)
            p2 = torch.cat(p2,1)

            if human_classifier.model_type == 'ternary':
                res_log_prob = human_classifier(torch.cat([p1, p2], 1))
                res_sel = res_log_prob.argmax(1)
                #if res_sel.item() == 0:
                #    res_sel = res_log_prob.argsort(1, descending=True)[0][1]

                if res_sel.item() == 0:
                    alg_darker = 'E'
                elif res_sel.item() == 2:
                    alg_darker = '1'
                else:
                    alg_darker = '2'
            elif human_classifier.model_type == 'binary':
                res_bin_prob = human_classifier(torch.cat([p1,p2], 1))

                if res_bin_prob[0,0] > eq_threshold:
                    alg_darker = 'E'
                else:
                    if res_bin_prob[0,1] > 0:
                        alg_darker = '1'
                    else:
                        alg_darker = '2'
            elif human_classifier.model_type == 'single_score':
                res_log_prob = human_classifier(torch.cat([p1, p2],1))
                res_sel = res_log_prob.argmax(1)
                #if res_sel.item() == 0:
                #    res_sel = res_log_prob.argsort(1, descending=True)[0][1]

                if res_sel.item() == 0:
                    alg_darker = 'E'
                elif res_sel.item() == 2:
                    alg_darker = '1'
                else:
                    alg_darker = '2'

            #print alg_darker, darker

            if darker == 'E':
                if darker != alg_darker:
                    error_equal_sum += weight

                weight_equal_sum += weight
            else:
                if darker != alg_darker:
                    error_inequal_sum += weight

                weight_inequal_sum += weight

            if darker != alg_darker:
                error_sum += weight

            if alg_darker == 'E':
                counts[0] += 1
            elif alg_darker == '1':
                counts[1] += 1
            elif alg_darker == '2':
                counts[2] += 1

            if darker == 'E':
                gt_counts[0] += 1
            elif darker == '1':
                gt_counts[1] += 1
            elif darker == '2':
                gt_counts[2] += 1

            weight_sum += weight

        print counts, gt_counts

        if weight_sum:
            return (error_sum / weight_sum), error_equal_sum/( weight_equal_sum + 1e-10), error_inequal_sum/(weight_inequal_sum + 1e-10)
        else:
            return None

    def compute_whdr(self, reflectance, judgements, delta=0.1):
        points = judgements['intrinsic_points']
        comparisons = judgements['intrinsic_comparisons']
        id_to_points = {p['id']: p for p in points}
        rows, cols = reflectance.shape[0:2]

        error_sum = 0.0
        error_equal_sum = 0.0
        error_inequal_sum = 0.0

        weight_sum = 0.0
        weight_equal_sum = 0.0
        weight_inequal_sum = 0.0

        for c in comparisons:
            # "darker" is "J_i" in our paper
            darker = c['darker']
            if darker not in ('1', '2', 'E'):
                continue

            # "darker_score" is "w_i" in our paper
            weight = c['darker_score']
            if weight <= 0.0 or weight is None:
                continue

            point1 = id_to_points[c['point1']]
            point2 = id_to_points[c['point2']]
            if not point1['opaque'] or not point2['opaque']:
                continue

            # convert to grayscale and threshold
            l1 = max(1e-10, np.mean(reflectance[
                int(point1['y'] * rows), int(point1['x'] * cols), ...]))
            l2 = max(1e-10, np.mean(reflectance[
                int(point2['y'] * rows), int(point2['x'] * cols), ...]))

            # # convert algorithm value to the same units as human judgements
            if l2 / l1 > 1.0 + delta:
                alg_darker = '1'
            elif l1 / l2 > 1.0 + delta:
                alg_darker = '2'
            else:
                alg_darker = 'E'

            if darker == 'E':
                if darker != alg_darker:
                    error_equal_sum += weight

                weight_equal_sum += weight
            else:
                if darker != alg_darker:
                    error_inequal_sum += weight

                weight_inequal_sum += weight

            if darker != alg_darker:
                error_sum += weight

            weight_sum += weight

        if weight_sum:
            return (error_sum / weight_sum), error_equal_sum/( weight_equal_sum + 1e-10), error_inequal_sum/(weight_inequal_sum + 1e-10)
        else:
            return None

    def evaluate_WHDR(self, prediction_R, targets, thresholds = [.1], human_classifier=None):
        # num_images = prediction_S.size(0) # must be even number
        total_whdr = np.zeros(len(thresholds))
        total_whdr_eq = np.zeros(len(thresholds))
        total_whdr_ineq = np.zeros(len(thresholds))

        count = float(0)

        for i in range(0, prediction_R.size(0)):
            prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
            if human_classifier is None:
                prediction_R_np = np.transpose(np.exp(prediction_R_np), (1,2,0))
            else:
                prediction_R_np = np.transpose(prediction_R_np, (1,2,0))

            o_h = targets['oringinal_shape'][0].numpy()
            o_w = targets['oringinal_shape'][1].numpy()
            # resize to original resolution
            #prediction_R_np = resize(prediction_R_np, (o_h[i],o_w[i]), order=1, preserve_range=True)

            # print(targets["judgements_path"][i])
            # load Json judgement
            judgements = json.load(open(targets["judgements_path"][i]))
            whdrs = []
            whdrs_eq = []
            whdrs_ineq = []

            if human_classifier is None:
                for t in thresholds:
                    whdr, whdr_eq, whdr_ineq = self.compute_whdr(prediction_R_np, judgements, t)
                    whdrs.append(whdr)
                    whdrs_eq.append(whdr_eq)
                    whdrs_ineq.append(whdr_ineq)
            elif human_classifier.model_type == 'binary':
                for t in thresholds:
                    if t == 0:
                        eq_threshold = -10000
                    elif t == 1:
                        eq_threshold = 10000
                    else:
                        eq_threshold = math.log(t/(1-t))

                    whdr, whdr_eq, whdr_ineq = self.compute_whdr_classifier(prediction_R_np, judgements, human_classifier, eq_threshold)
                    whdrs.append(whdr)
                    whdrs_eq.append(whdr_eq)
                    whdrs_ineq.append(whdr_ineq)
            else:
                eq_threshold = 0.0

                whdr, whdr_eq, whdr_ineq = self.compute_whdr_classifier(prediction_R_np, judgements, human_classifier, eq_threshold)
                whdrs.append(whdr)
                whdrs_eq.append(whdr_eq)
                whdrs_ineq.append(whdr_ineq)

            total_whdr += np.array(whdrs)
            total_whdr_eq += np.array(whdrs_eq)
            total_whdr_ineq += np.array(whdrs_ineq)
            count += 1.

        return total_whdr, total_whdr_eq, total_whdr_ineq, count



    def evaluate_RC_loss(self, prediction_n, targets):

        normal_norm = torch.sqrt( torch.sum(torch.pow(prediction_n , 2) , 1) )
        normal_norm = normal_norm.unsqueeze(1).repeat(1,3,1,1)
        prediction_n = torch.div(prediction_n , normal_norm)

        # mask_0 = Variable(targets['mask'].cuda(), requires_grad = False)
        # n_gt_0 = Variable(targets['normal'].cuda(), requires_grad = False)

        total_loss = self.AngleLoss(prediction_n, targets)

        return total_loss.data[0]

    def evaluate_L0_loss(self, prediction_R, targets):
        # num_images = prediction_S.size(0) # must be even number
        total_whdr = float(0)
        count = float(0)

        for i in range(0, 1):
            prediction_R_np = prediction_R
            # prediction_R_np = prediction_R.data[i,:,:,:].cpu().numpy()
            # prediction_R_np = np.transpose(prediction_R_np, (1,2,0))

            # load Json judgement
            judgements = json.load(open(targets["judgements_path"][i]))
            whdr = self.compute_whdr(prediction_R_np, judgements, 0.1)

            total_whdr += whdr
            count += 1

        return total_whdr, count


    def get_loss_var(self):
        return self.total_loss


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            n_output_dim = 3
            uprelu1 = nn.ReLU(False)
            uprelu2 = nn.ReLU(False)
            upconv_1 = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            upconv_2 = nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)

            conv_1 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1)
            conv_2 = nn.Conv2d(inner_nc, inner_nc, kernel_size=3,
                             stride=1, padding=1)

            # conv_1_o = nn.Conv2d(inner_nc, 1, kernel_size=3,
                             # stride=1, padding=1)
            conv_2_o = nn.Conv2d(inner_nc, n_output_dim, kernel_size=3,
                             stride=1, padding=1)

            upnorm_1 = norm_layer(inner_nc, affine=True)
            upnorm_2 = norm_layer(inner_nc, affine=True)
            # uprelu2_o = nn.ReLU(False)

            down = [downconv]
            up_1 = [uprelu1, upconv_1, upnorm_1, nn.ReLU(False), conv_1, nn.ReLU(False), conv_1_o]
            up_2 = [uprelu2, upconv_2, upnorm_2, nn.ReLU(False), conv_2, nn.ReLU(False), conv_2_o]

            self.downconv_model = nn.Sequential(*down)
            self.upconv_model_1 = nn.Sequential(*up_1)
            self.upconv_model_2 = nn.Sequential(*up_2)
            self.submodule = submodule

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            self.model = nn.Sequential(*model)

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.model = nn.Sequential(*model)

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            # return self.model(x)
            down_x = self.downconv_model(x)
            y = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y)
            y_2 = self.upconv_model_2(y)

            return y_1, y_2

        else:
            return torch.cat([self.model(x), x], 1)


class SingleUnetGenerator_S(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SingleUnetGenerator_S, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = SingleUnetSkipConnectionBlock_S(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = SingleUnetSkipConnectionBlock_S(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = SingleUnetSkipConnectionBlock_S(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_S(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_S(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_S(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


class SingleUnetSkipConnectionBlock_S(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SingleUnetSkipConnectionBlock_S, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, 1,
                                        kernel_size=4, stride=2,
                                        padding=1)

            down = [downconv]
            up = [uprelu, upconv]
            model = down + [submodule]
            self.model = nn.Sequential(*model)
            self.up_model = nn.Sequential(*up)

        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            # model = down + up

            int_conv = [nn.AdaptiveAvgPool2d((2,2)), nn.Conv2d(inner_nc, inner_nc/2, kernel_size=3, stride=2, padding=1), nn.ReLU(False)]

            fc = [nn.Linear(256, 3)]
            self.int_conv = nn.Sequential(* int_conv)
            self.fc = nn.Sequential(* fc)

            self.down_model = nn.Sequential(*down)
            self.up_model = nn.Sequential(*up)

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] #+ up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] # + up

            if use_dropout:
                upconv_model = up + [nn.Dropout(0.5)]
            else:
                upconv_model = up

            self.model = nn.Sequential(*model)
            self.up_model = nn.Sequential(*upconv_model)

    def forward(self, x):

        if self.outermost:
            y_1, color_s = self.model(x)
            y_1 = self.up_model(y_1)

            return y_1, color_s
        elif self.innermost:
            y_1 = self.down_model(x)
            color_s = self.int_conv(y_1)
            color_s = color_s.view(color_s.size(0), -1)
            color_s  = self.fc(color_s)

            y_1 = self.up_model(y_1)
            y_1 = torch.cat([y_1, x], 1)

            return y_1, color_s
        else:
            y_1, color_s = self.model(x)
            y_1 = self.up_model(y_1)

            return torch.cat([y_1, x], 1), color_s



class SingleUnetGenerator_R(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SingleUnetGenerator_R, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = SingleUnetSkipConnectionBlock_R(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = SingleUnetSkipConnectionBlock_R(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = SingleUnetSkipConnectionBlock_R(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_R(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_R(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_R(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SingleUnetSkipConnectionBlock_R(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SingleUnetSkipConnectionBlock_R, self).__init__()
        self.outermost = outermost

        if outermost:
            downconv = nn.Conv2d(3, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        else:
            downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        num_output = outer_nc

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, num_output,
                                        # kernel_size=4, stride=2,
                                        # padding=1)

            upconv = [uprelu, nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2, padding=1), nn.ReLU(False),
                                        nn.Conv2d(inner_nc, num_output, kernel_size=1)]

            down = [downconv]
            up = upconv
            model = down + [submodule] + up
            self.model = nn.Sequential(*model)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
            self.model = nn.Sequential(*model)

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up
            self.model = nn.Sequential(*model)

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


class SingleUnetGenerator_L(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(SingleUnetGenerator_L, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = SingleUnetSkipConnectionBlock_L(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = SingleUnetSkipConnectionBlock_L(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)

        unet_block = SingleUnetSkipConnectionBlock_L(ngf * 4, ngf * 8, unet_block, gird = True, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_L(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_L(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = SingleUnetSkipConnectionBlock_L(output_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class SingleUnetSkipConnectionBlock_L(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, gird =False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(SingleUnetSkipConnectionBlock_L, self).__init__()
        self.outermost = outermost
        self.gird = grid
        if outermost:
            downconv = nn.Conv2d(3, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        else:
            downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)

        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc, affine=True)

        num_output = outer_nc

        if outermost:
            # upconv = nn.ConvTranspose2d(inner_nc * 2, num_output,
                                        # kernel_size=4, stride=2,
                                        # padding=1)

            upconv = [uprelu, nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2, padding=1), nn.ReLU(False),
                                        nn.Conv2d(inner_nc, 1, kernel_size=1), nn.Sigmoid()]

            down = [downconv]
            up = upconv
            model = down + [submodule] + up
            self.model = nn.Sequential(*model)
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            model = down + up
            self.model = nn.Sequential(*model)

        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

            if self.gird:
                grid_layer = [nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(inner_nc, affine=True), nn.ReLU(False),
                                nn.Conv2d(inner_nc, inner_nc/4, kernel_size=3, padding=1), nn.ReLU(False),
                                nn.Conv2d(inner_nc/4, num_output, kernel_size=1)]
                self.grid_layer = nn.Sequential(*grid_layer)

            self.model = nn.Sequential(*model)

        # self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            y = self.model(x)
            return y, self.grid_y
        else:
            y = self.model(x)

            if self.grid:
                upsample_layer = nn.Upsample(scale_factor= 8, mode='bilinear')
                self.grid_y = upsample_layer(self.grid_layer(y))

            return torch.cat([y, x], 1)


class MultiUnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(MultiUnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        # assert(input_nc == output_nc)

        # construct unet structure
        unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, innermost=True)
        for i in range(num_downs - 5):
            unet_block = MultiUnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer)
        unet_block = MultiUnetSkipConnectionBlock(input_nc, ngf, unet_block, outermost=True, norm_layer=norm_layer, output_nc = output_nc)

        self.model = unet_block

    def forward(self, input):
        if  self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
            # self.model(input)

# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class MultiUnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False, output_nc=None):
        super(MultiUnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        # print("we are in mutilUnet")
        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc, affine=True)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc, affine=True)

        if outermost:
            if output_nc is None:
                output_nc = outer_nc

            down = [downconv]

            upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2, padding=1), norm_layer(inner_nc, affine=True), nn.ReLU(False),
                                nn.Conv2d(inner_nc, output_nc, kernel_size= 1, bias=True)]
            upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, inner_nc,
                                        kernel_size=4, stride=2, padding=1) , norm_layer(inner_nc, affine=True), nn.ReLU(False),
                                nn.Conv2d(inner_nc, 1, kernel_size= 1, bias=True)]
        elif innermost:
            down = [downrelu, downconv]
            upconv_model_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            upconv_model_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
        else:
            down = [downrelu, downconv, downnorm]
            up_1 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]
            up_2 = [nn.ReLU(False), nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1), norm_layer(outer_nc, affine=True)]

            if use_dropout:
                upconv_model_1 = up_1 + [nn.Dropout(0.5)]
                upconv_model_2 = up_2 + [nn.Dropout(0.5)]
                # model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                upconv_model_1 = up_1
                upconv_model_2 = up_2

            # model = down + [submodule]

        self.downconv_model = nn.Sequential(*down)
        self.submodule = submodule
        self.upconv_model_1 = nn.Sequential(*upconv_model_1)
        self.upconv_model_2 = nn.Sequential(*upconv_model_2)

    def forward(self, x):

        if self.outermost:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            # y_u = self.upconv_model_u(y_1)
            y_1 = self.upconv_model_1(y_1)

            y_2 = self.upconv_model_2(y_2)

            return y_1, y_2
            # return self.model(x)
        elif self.innermost:
            down_output = self.downconv_model(x)

            y_1 = self.upconv_model_1(down_output)
            y_2 = self.upconv_model_2(down_output)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2
        else:
            down_x = self.downconv_model(x)
            y_1, y_2 = self.submodule.forward(down_x)
            y_1 = self.upconv_model_1(y_1)
            y_2 = self.upconv_model_2(y_2)
            y_1 = torch.cat([y_1, x], 1)
            y_2 = torch.cat([y_2, x], 1)

            return y_1, y_2

