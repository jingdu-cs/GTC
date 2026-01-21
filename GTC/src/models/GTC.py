import os
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, degree

from common.abstract_recommender import GeneralRecommender
from utils.diffusion import Diffusion
from utils.corr import Total_Correlation


class GTC(GeneralRecommender):
    def __init__(self, config, dataset):
        super(GTC, self).__init__(config, dataset)

        num_user = self.n_users
        num_item = self.n_items
        batch_size = config['train_batch_size']  # not used
        dim_x = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        has_id = True

        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.k = 40
        self.aggr_mode = config['aggr_mode']
        self.user_aggr_mode = 'softmax'
        self.num_layer = 1
        self.cold_start = 0
        self.dataset = dataset
        self.construction = config['construction']
        self.reg_weight = config['reg_weight']
        self.drop_rate = 0.1
        self.dim_latent = 64
        self.mm_adj = None
        self.v_mlp = nn.Linear(self.v_feat.shape[1], self.dim_latent)
        self.t_mlp = nn.Linear(self.t_feat.shape[1], self.dim_latent)

        self.t_diffusion = Diffusion(noise_steps=config['noise_steps'], beta_end=config['beta_end'], device=self.device)
        self.v_diffusion = Diffusion(noise_steps=config['noise_steps'], beta_end=config['beta_end'], device=self.device)
        self.diffusion_loss = nn.MSELoss()
        self.diff_weight = config["diff_weight"]
        self.tc_loss = Total_Correlation()
        self.symile_weight = config["symile_weight"]
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        self.user_graph_dict = np.load(os.path.join(dataset_path, config['user_graph_dict_file']),
                                       allow_pickle=True).item()

        mm_adj_file = os.path.join(dataset_path, 'mm_adj_{}.pt'.format(self.knn_k))

        self.item_embedding = nn.Embedding(self.num_item, dim_x).to(self.device)
        self.id_rep = self.item_embedding.weight
        if self.v_feat is not None:
            print("v_feat exists!")
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            # self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            print("t_feat exists!")
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            # self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)

        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                indices, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                indices, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj
                del image_adj
            torch.save(self.mm_adj, mm_adj_file)

        # packing interaction in training into edge_index
        train_interactions = dataset.inter_matrix(form='coo').astype(np.float32)
        edge_index = self.pack_edge_index(train_interactions)
        self.edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(self.device)
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1, 0]]), dim=1)

        # Attention weights for user multi-modal fusion
        self.weight_u = nn.Parameter(nn.init.xavier_normal_(
            torch.tensor(np.random.randn(self.num_user, 3, 1), dtype=torch.float32, requires_grad=True)))
        self.weight_u.data = F.softmax(self.weight_u, dim=1)

        self.item_index = torch.zeros([self.num_item], dtype=torch.long)
        index = []
        for i in range(self.num_item):
            self.item_index[i] = i
            index.append(i)
        self.drop_percent = self.drop_rate
        self.single_percent = 1
        self.double_percent = 0

        edge_index = edge_index[np.lexsort(edge_index.T[1, None])]

        self.MLP_user = nn.Linear(self.dim_latent * 2, self.dim_latent)

        self.i_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                         num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                         device=self.device, features=self.id_rep)
        
        if self.v_feat is not None:
            dummy_v_feat = torch.zeros(self.num_item, self.dim_latent)
            self.v_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                             device=self.device, features=dummy_v_feat)
            del dummy_v_feat

        if self.t_feat is not None:
            dummy_t_feat = torch.zeros(self.num_item, self.dim_latent)
            self.t_gcn = GCN(self.dataset, batch_size, num_user, num_item, dim_x, self.aggr_mode,
                             num_layer=self.num_layer, has_id=has_id, dropout=self.drop_rate, dim_latent=None,
                             device=self.device, features=dummy_t_feat)
            del dummy_t_feat
        self.user_graph = User_Graph_sample(num_user, 'add', self.dim_latent)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device)
        indices0 = torch.unsqueeze(indices0, 1)
        indices0 = indices0.expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)
        # norm
        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt
        return torch.sparse.FloatTensor(indices, values, adj_size)

    def pre_epoch_processing(self):
        self.epoch_user_graph, self.user_weight_matrix = self.topk_sample(self.k)
        self.user_weight_matrix = self.user_weight_matrix.to(self.device)

    def cross_modal_refinement(self, id_feat, v_feat, t_feat, temperature=1.0):
        mip_id_v = (id_feat * v_feat).sum(dim=1, keepdim=True)
        mip_id_t = (id_feat * t_feat).sum(dim=1, keepdim=True)
        mip_v_t = (v_feat * t_feat).sum(dim=1, keepdim=True)

        id_similarities = torch.cat([mip_id_v, mip_id_t, mip_v_t], dim=1)
        id_weights = F.softmax(id_similarities / temperature, dim=1)

        id_refined = (id_weights[:, 0:1] * id_feat + 
                    id_weights[:, 1:2] * v_feat + 
                    id_weights[:, 2:3] * t_feat)
        id_final = id_feat + id_refined
        return id_final

    def pack_edge_index(self, inter_mat):
        rows = inter_mat.row
        cols = inter_mat.col + self.n_users
        return np.column_stack((rows, cols))

    def forward(self, interaction):
        user_nodes, pos_item_nodes, neg_item_nodes = interaction[0], interaction[1], interaction[2]
        pos_item_nodes += self.n_users
        neg_item_nodes += self.n_users
        representation = None

        if self.v_feat is None and self.t_feat is None:
            self.i_rep, self.i_preference = self.i_gcn(self.edge_index, self.id_rep)
            self.item_rep = self.i_rep[self.num_user:]
            self.user_rep = self.i_rep[:self.num_user]
            self.result_embed = torch.cat((self.user_rep, self.item_rep), dim=0)
            user_tensor = self.result_embed[user_nodes]
            pos_item_tensor = self.result_embed[pos_item_nodes]
            neg_item_tensor = self.result_embed[neg_item_nodes]
            pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
            neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

            return pos_scores, neg_scores
        else:
            v_feat = self.v_mlp(self.v_feat)
            t_feat = self.t_mlp(self.t_feat)

            if self.training:
                t = self.t_diffusion.sample_timesteps(self.id_rep.shape[0]).to(self.device)
            else:
                t = torch.ones(self.id_rep.shape[0], dtype=torch.long).to(self.device) * (self.t_diffusion.noise_steps // 2)
            
            xt_v, v_noise = self.v_diffusion.noise_images(v_feat, t, self.id_rep)
            xt_t, t_noise = self.t_diffusion.noise_images(t_feat, t, self.id_rep)
            
            v_predict_noise = self.v_diffusion.denosing(xt_v, t, self.id_rep)
            t_predict_noise = self.t_diffusion.denosing(xt_t, t, self.id_rep)
            
            self.gen_loss = self.diffusion_loss(v_noise, v_predict_noise) + self.diffusion_loss(t_noise, t_predict_noise)

            alpha_hat_t = self.v_diffusion.alpha_hat[t][:, None]
            sqrt_alpha_hat_t = torch.sqrt(alpha_hat_t)
            sqrt_one_minus_alpha_hat_t = torch.sqrt(1 - alpha_hat_t)
            
            denoise_v = (xt_v - sqrt_one_minus_alpha_hat_t * v_predict_noise) / sqrt_alpha_hat_t
            denoise_t = (xt_t - sqrt_one_minus_alpha_hat_t * t_predict_noise) / sqrt_alpha_hat_t

            reconstruct_rep = self.cross_modal_refinement(self.id_rep, denoise_v, denoise_t)
            self.inter_rep = reconstruct_rep

            self.i_rep_norm = F.normalize(self.id_rep, p=2.0, dim=1)
            self.v_rep_norm = F.normalize(denoise_v, p=2.0, dim=1)
            self.t_rep_norm = F.normalize(denoise_t, p=2.0, dim=1)

            self.tot_corr_loss = torch.tensor(0.0, requires_grad=True)
            logit_scale_exp = self.logit_scale.exp()
            self.tot_corr_loss = self.tc_loss(
                [self.i_rep_norm, self.v_rep_norm, self.t_rep_norm], logit_scale_exp)

            self.i_rep, self.i_preference = self.i_gcn(self.edge_index, self.inter_rep)
            self.v_rep, self.v_preference = self.v_gcn(self.edge_index, denoise_v)
            self.t_rep, self.t_preference = self.t_gcn(self.edge_index, denoise_t)

            if self.construction == 'cat' or self.construction == 'diffusion':
                if self.v_feat is not None:
                    representation = torch.cat((self.i_rep, self.v_rep), dim=1)
                if self.t_feat is not None:
                    representation = torch.cat((self.i_rep, self.t_rep), dim=1)
                if self.v_feat is not None and self.t_feat is not None:
                    representation = torch.cat((self.i_rep, self.v_rep, self.t_rep), dim=1)
            else:
                representation = self.i_rep + self.t_rep + self.v_rep

            item_rep = representation[self.num_user:]

            if self.v_feat is not None and self.t_feat is None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.i_rep = torch.unsqueeze(self.i_rep, 2)
                user_rep = torch.cat((self.i_rep[:self.num_user], self.v_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep
                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

            if self.t_feat is not None and self.v_feat is None:
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                self.i_rep = torch.unsqueeze(self.i_rep, 2)
                user_rep = torch.cat((self.i_rep[:self.num_user], self.t_rep[:self.num_user]), dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep
                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1]), dim=1)

            if self.v_feat is not None and self.t_feat is not None:
                self.v_rep = torch.unsqueeze(self.v_rep, 2)
                self.t_rep = torch.unsqueeze(self.t_rep, 2)
                self.i_rep = torch.unsqueeze(self.i_rep, 2)
                user_rep = torch.cat(
                    [self.i_rep[:self.num_user], self.v_rep[:self.num_user], self.t_rep[:self.num_user]], dim=2)
                user_rep = self.weight_u.transpose(1, 2) * user_rep
                user_rep = torch.cat((user_rep[:, :, 0], user_rep[:, :, 1], user_rep[:, :, 2]), dim=1)

            h = item_rep
            for i in range(self.n_layers):
                h = torch.sparse.mm(self.mm_adj, h)
            h_u1 = self.user_graph(user_rep, self.epoch_user_graph, self.user_weight_matrix)
            self.user_rep = user_rep + h_u1
            self.item_rep = item_rep + h
            self.result_embed = torch.cat((self.user_rep, self.item_rep), dim=0)
            user_tensor = self.result_embed[user_nodes]
            pos_item_tensor = self.result_embed[pos_item_nodes]
            neg_item_tensor = self.result_embed[neg_item_nodes]
            pos_scores = torch.sum(user_tensor * pos_item_tensor, dim=1)
            neg_scores = torch.sum(user_tensor * neg_item_tensor, dim=1)

            return pos_scores, neg_scores

    def calculate_loss(self, interaction):
        user = interaction[0]
        pos_scores, neg_scores = self.forward(interaction)
        loss_value = -torch.mean(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        
        if self.v_feat is None and self.t_feat is None:
            reg_embedding_loss_i = (self.i_rep[user] ** 2).mean() if self.i_rep is not None else 0.0
            reg_loss = self.reg_weight * reg_embedding_loss_i
            return loss_value + reg_loss

        reg_embedding_loss_v = (self.v_rep[user] ** 2).mean() if self.v_rep is not None else 0.0
        reg_embedding_loss_t = (self.t_rep[user] ** 2).mean() if self.t_rep is not None else 0.0
        reg_embedding_loss_i = (self.i_rep[user] ** 2).mean() if self.i_rep is not None else 0.0

        reg_loss = self.reg_weight * (reg_embedding_loss_v + reg_embedding_loss_t + reg_embedding_loss_i)
        reg_loss += self.reg_weight * (self.weight_u ** 2).mean()
        return loss_value + reg_loss + self.symile_weight * self.tot_corr_loss + self.diff_weight * self.gen_loss

    def full_sort_predict(self, interaction):
        user_tensor = self.result_embed[:self.n_users]
        item_tensor = self.result_embed[self.n_users:]

        temp_user_tensor = user_tensor[interaction[0], :]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())
        return score_matrix

    def topk_sample(self, k):
        user_graph_index = []
        count_num = 0
        user_weight_matrix = torch.zeros(len(self.user_graph_dict), k)
        tasike = []
        for i in range(k):
            tasike.append(0)
        for i in range(len(self.user_graph_dict)):
            if len(self.user_graph_dict[i][0]) < k:
                count_num += 1
                if len(self.user_graph_dict[i][0]) == 0:
                    # pdb.set_trace()
                    user_graph_index.append(tasike)
                    continue
                user_graph_sample = self.user_graph_dict[i][0][:k]
                user_graph_weight = self.user_graph_dict[i][1][:k]
                while len(user_graph_sample) < k:
                    rand_index = np.random.randint(0, len(user_graph_sample))
                    user_graph_sample.append(user_graph_sample[rand_index])
                    user_graph_weight.append(user_graph_weight[rand_index])
                user_graph_index.append(user_graph_sample)

                if self.user_aggr_mode == 'softmax':
                    user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
                if self.user_aggr_mode == 'mean':
                    user_weight_matrix[i] = torch.ones(k) / k  # mean
                continue
            user_graph_sample = self.user_graph_dict[i][0][:k]
            user_graph_weight = self.user_graph_dict[i][1][:k]

            if self.user_aggr_mode == 'softmax':
                user_weight_matrix[i] = F.softmax(torch.tensor(user_graph_weight), dim=0)  # softmax
            if self.user_aggr_mode == 'mean':
                user_weight_matrix[i] = torch.ones(k) / k  # mean
            user_graph_index.append(user_graph_sample)

        # pdb.set_trace()
        return user_graph_index, user_weight_matrix

class User_Graph_sample(torch.nn.Module):
    def __init__(self, num_user, aggr_mode, dim_latent):
        super(User_Graph_sample, self).__init__()
        self.num_user = num_user
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode

    def forward(self, features, user_graph, user_matrix):
        index = user_graph
        u_features = features[index]
        user_matrix = user_matrix.unsqueeze(1)
        # pdb.set_trace()
        u_pre = torch.matmul(user_matrix, u_features)
        u_pre = u_pre.squeeze()
        return u_pre


class GCN(torch.nn.Module):
    def __init__(self, datasets, batch_size, num_user, num_item, dim_id, aggr_mode, num_layer, has_id, dropout,
                 dim_latent=None, device=None, features=None):
        """
        Graph Convolutional Network for multi-modal recommendation.
        
        Args:
            dim_latent: If None, use features as-is (for 64-dim features).
                        If specified, project features to this dimension via MLP.
            features: Only used to infer input dimension (not stored).
        """
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.datasets = datasets
        self.dim_id = dim_id
        self.dim_feat = features.size(1) if features is not None else 64
        self.dim_latent = dim_latent
        self.aggr_mode = aggr_mode
        self.num_layer = num_layer
        self.has_id = has_id
        self.dropout = dropout
        self.device = device

        output_dim = self.dim_latent if self.dim_latent else self.dim_feat
        
        self.preference = nn.Parameter(nn.init.xavier_normal_(torch.tensor(
            np.random.randn(self.num_user, output_dim), dtype=torch.float32, requires_grad=True),
            gain=1).to(self.device))
        
        if self.dim_latent:
            self.MLP = nn.Linear(self.dim_feat, 4 * self.dim_latent)
            self.MLP_1 = nn.Linear(4 * self.dim_latent, self.dim_latent)

        self.conv_embed_1 = Base_gcn(output_dim, output_dim, aggr=self.aggr_mode)

    def forward(self, edge_index, features):
        if self.dim_latent:
            temp_features = self.MLP_1(F.leaky_relu(self.MLP(features)))
        else:
            temp_features = features
        
        x = torch.cat((self.preference, temp_features), dim=0).to(self.device)
        x = F.normalize(x).to(self.device)
        
        h = self.conv_embed_1(x, edge_index)
        h_1 = self.conv_embed_1(h, edge_index)
        
        x_hat = h + x + h_1
        
        return x_hat, self.preference


class Base_gcn(MessagePassing):
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(Base_gcn, self).__init__(aggr=aggr, **kwargs)
        self.aggr = aggr
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index, size=None):
        if size is None:
            edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_j, edge_index, size):
        if self.aggr == 'add':
            row, col = edge_index
            deg = degree(row, size[0], dtype=x_j.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            return norm.view(-1, 1) * x_j
        return x_j

    def update(self, aggr_out):
        return aggr_out

    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
