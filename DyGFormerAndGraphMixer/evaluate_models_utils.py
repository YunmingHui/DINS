import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import random

from utils.utils import NegativeEdgeSampler, NeighborSampler
from utils.DataLoader import Data
from sklearn.metrics import average_precision_score, roc_auc_score


def evaluate_model_link_prediction(model_name: str, model: nn.Module, neighbor_sampler: NeighborSampler, evaluate_idx_data_loader: DataLoader,
                                   evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate_data: Data, loss_func: nn.Module,
                                   num_neighbors: int = 20, time_gap: int = 2000):
    """
    evaluate models on the link prediction task
    :param model_name: str, name of the model
    :param model: nn.Module, the model to be evaluated
    :param neighbor_sampler: NeighborSampler, neighbor sampler
    :param evaluate_idx_data_loader: DataLoader, evaluate index data loader
    :param evaluate_neg_edge_sampler: NegativeEdgeSampler, evaluate negative edge sampler
    :param evaluate_data: Data, data to be evaluated
    :param loss_func: nn.Module, loss function
    :param num_neighbors: int, number of neighbors to sample for each node
    :param time_gap: int, time gap for neighbors to compute node features
    :return:
    """
    # Ensures the random sampler uses a fixed seed for evaluation (i.e. we always sample the same negatives for validation / test set)
    assert evaluate_neg_edge_sampler.seed is not None
    evaluate_neg_edge_sampler.reset_random_state()

    model[0].set_neighbor_sampler(neighbor_sampler)

    original_sample_way = evaluate_neg_edge_sampler.negative_sample_strategy
    evaluate_neg_edge_sampler.negative_sample_strategy = 'random'

    model.eval()

    random_sender_ap, random_sender_auc, random_receiver_ap, random_receiver_auc,  = [], [], [], []
    hist_6h_ap, hist_6h_auc, hist_12h_ap, hist_12h_auc, hist_24h_ap, hist_24h_auc = [], [], [], [], [], []
    loop_ap, loop_auc = [], []

    overall_predict = np.array([[]])
    overall_label = np.array([])

    all_nodes = set()
    loop_nodes = set()
    for i in range(len(evaluate_data.src_node_ids)):
        if evaluate_data.src_node_ids[i] == evaluate_data.dst_node_ids[i]:
            loop_nodes.add(evaluate_data.src_node_ids[i])
        all_nodes.add(evaluate_data.src_node_ids[i])
        all_nodes.add(evaluate_data.dst_node_ids[i])


    with torch.no_grad():
        # store evaluate losses and metrics
        for evaluate_data_indices in evaluate_idx_data_loader:
            evaluate_data_indices = evaluate_data_indices.numpy()
            batch_src_node_ids, batch_dst_node_ids, batch_node_interact_times, batch_edge_ids = \
                evaluate_data.src_node_ids[evaluate_data_indices],  evaluate_data.dst_node_ids[evaluate_data_indices], \
                evaluate_data.node_interact_times[evaluate_data_indices], evaluate_data.edge_ids[evaluate_data_indices]
            
            batch_src_loop_node_ids, batch_dst_loop_node_ids, batch_loop_node_interact_times = [], [], []
            batch_src_no_loop_node_ids, batch_dst_no_loop_node_ids, batch_no_loop_node_interact_times = [], [], []

            for i in range(len(batch_src_node_ids)):
                if batch_src_node_ids[i] == batch_dst_node_ids[i]:
                    batch_src_loop_node_ids.append(batch_src_node_ids[i])
                    batch_dst_loop_node_ids.append(batch_dst_node_ids[i])
                    batch_loop_node_interact_times.append(batch_node_interact_times[i])
                else:
                    batch_src_no_loop_node_ids.append(batch_src_node_ids[i])
                    batch_dst_no_loop_node_ids.append(batch_dst_node_ids[i])
                    batch_no_loop_node_interact_times.append(batch_node_interact_times[i])

            batch_src_loop_node_ids = np.array(batch_src_loop_node_ids)
            batch_dst_loop_node_ids = np.array(batch_dst_loop_node_ids)
            batch_loop_node_interact_times = np.array(batch_loop_node_interact_times)
            batch_src_no_loop_node_ids = np.array(batch_src_no_loop_node_ids)
            batch_dst_no_loop_node_ids = np.array(batch_dst_no_loop_node_ids)
            batch_no_loop_node_interact_times = np.array(batch_no_loop_node_interact_times)            

            batch_neg_src_node_ids, batch_neg_dst_node_ids = evaluate_neg_edge_sampler.sample(size=len(batch_dst_no_loop_node_ids))

            # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
            # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
            if model_name in ['GraphMixer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                
                batch_src_no_loop_node_embeddings, batch_dst_no_loop_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_no_loop_node_ids,
                                                                      dst_node_ids=batch_dst_no_loop_node_ids,
                                                                      node_interact_times=batch_no_loop_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_no_loop_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                

                # get temporal embedding of sampled non-loop nodes
                no_loop_nodes = all_nodes - loop_nodes
                sample_nodes = random.sample(no_loop_nodes, len(batch_src_node_ids))
                sample_nodes = np.array(sample_nodes)
                batch_neg_src_node_embeddings_loop, batch_neg_dst_node_embeddings_loop = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=sample_nodes,
                                                                      dst_node_ids=sample_nodes,
                                                                      node_interact_times=batch_node_interact_times,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                
                # get temporal embedding of negative source and negative destination nodes after 6 hours
                batch_neg_src_node_embeddings_6h, batch_neg_dst_node_embeddings_6h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+6*60/5,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                
                # get temporal embedding of negative source and negative destination nodes after 12 hours
                batch_neg_src_node_embeddings_12h, batch_neg_dst_node_embeddings_12h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+12*60/5,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)
                
                # get temporal embedding of negative source and negative destination nodes after 24 hours
                batch_neg_src_node_embeddings_24h, batch_neg_dst_node_embeddings_24h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+24*60/5,
                                                                      num_neighbors=num_neighbors,
                                                                      time_gap=time_gap)

            elif model_name in ['DyGFormer']:
                # get temporal embedding of source and destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_src_node_embeddings, batch_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times)
                
                batch_src_no_loop_node_embeddings, batch_dst_no_loop_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_no_loop_node_ids,
                                                                      dst_node_ids=batch_dst_no_loop_node_ids,
                                                                      node_interact_times=batch_no_loop_node_interact_times)

                # get temporal embedding of negative source and negative destination nodes
                # two Tensors, with shape (batch_size, node_feat_dim)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_neg_src_node_ids,
                                                                      dst_node_ids=batch_neg_dst_node_ids,
                                                                      node_interact_times=batch_no_loop_node_interact_times)
                
                # get temporal embedding of sampled non-loop nodes
                no_loop_nodes = all_nodes - loop_nodes
                sample_nodes = random.sample(no_loop_nodes, len(batch_src_node_ids))
                sample_nodes = np.array(sample_nodes)
                batch_neg_src_node_embeddings_loop, batch_neg_dst_node_embeddings_loop = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=sample_nodes,
                                                                      dst_node_ids=sample_nodes,
                                                                      node_interact_times=batch_node_interact_times)
                
                # get temporal embedding of negative source and negative destination nodes after 6 hours
                batch_neg_src_node_embeddings_6h, batch_neg_dst_node_embeddings_6h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+6*60/5)
                
                # get temporal embedding of negative source and negative destination nodes after 12 hours
                batch_neg_src_node_embeddings_12h, batch_neg_dst_node_embeddings_12h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+12*60/5)
                
                # get temporal embedding of negative source and negative destination nodes after 24 hours
                batch_neg_src_node_embeddings_24h, batch_neg_dst_node_embeddings_24h = \
                    model[0].compute_src_dst_node_temporal_embeddings(src_node_ids=batch_src_node_ids,
                                                                      dst_node_ids=batch_dst_node_ids,
                                                                      node_interact_times=batch_node_interact_times+24*60/5)
            else:
                raise ValueError(f"Wrong value for model_name {model_name}!")
            
            positive_probabilities_all = model[1](input_1=batch_src_node_embeddings, input_2=batch_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities_6h = model[1](input_1=batch_neg_src_node_embeddings_6h, input_2=batch_neg_dst_node_embeddings_6h).squeeze(dim=-1).sigmoid()
            negative_probabilities_12h = model[1](input_1=batch_neg_src_node_embeddings_12h, input_2=batch_neg_dst_node_embeddings_12h).squeeze(dim=-1).sigmoid()
            negative_probabilities_24h = model[1](input_1=batch_neg_src_node_embeddings_24h, input_2=batch_neg_dst_node_embeddings_24h).squeeze(dim=-1).sigmoid()
            negative_probabilities_loop = model[1](input_1=batch_neg_src_node_embeddings_loop, input_2=batch_neg_dst_node_embeddings_loop).squeeze(dim=-1).sigmoid()

            if overall_predict.size == 0:
                overall_predict = positive_probabilities_all.cpu().detach().numpy()
            else:
                overall_predict = np.concatenate([overall_predict, positive_probabilities_all.cpu().detach().numpy()])

            overall_predict = np.concatenate([overall_predict, negative_probabilities_6h.cpu().detach().numpy(), negative_probabilities_12h.cpu().detach().numpy(), negative_probabilities_24h.cpu().detach().numpy(), negative_probabilities_loop.cpu().detach().numpy()])
            overall_label = np.concatenate([overall_label, np.ones(len(positive_probabilities_all)), np.zeros(len(negative_probabilities_6h)), np.zeros(len(negative_probabilities_12h)), np.zeros(len(negative_probabilities_24h)), np.zeros(len(negative_probabilities_loop))])
            
            labels = np.concatenate([np.ones(len(positive_probabilities_all)), np.zeros(len(negative_probabilities_6h))])
            
            predicts = torch.cat([positive_probabilities_all, negative_probabilities_6h], dim=0).cpu().detach().numpy()
            hist_6h_ap.append(average_precision_score(labels, predicts))
            hist_6h_auc.append(roc_auc_score(labels, predicts))

            predicts = torch.cat([positive_probabilities_all, negative_probabilities_12h], dim=0).cpu().detach().numpy()
            hist_12h_ap.append(average_precision_score(labels, predicts))
            hist_12h_auc.append(roc_auc_score(labels, predicts))

            predicts = torch.cat([positive_probabilities_all, negative_probabilities_24h], dim=0).cpu().detach().numpy()
            hist_24h_ap.append(average_precision_score(labels, predicts))
            hist_24h_auc.append(roc_auc_score(labels, predicts))
            
            predicts = torch.cat([positive_probabilities_all, negative_probabilities_loop], dim=0).cpu().detach().numpy()
            loop_ap.append(average_precision_score(labels, predicts))
            loop_auc.append(roc_auc_score(labels, predicts))


            positive_probabilities_non_loop = model[1](input_1=batch_src_no_loop_node_embeddings, input_2=batch_dst_no_loop_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities_swap = model[1](input_1=batch_dst_no_loop_node_embeddings, input_2=batch_src_no_loop_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities_random_recevier = model[1](input_1=batch_src_no_loop_node_embeddings, input_2=batch_neg_dst_node_embeddings).squeeze(dim=-1).sigmoid()
            negative_probabilities_random_sender = model[1](input_1=batch_neg_src_node_embeddings, input_2=batch_dst_no_loop_node_embeddings).squeeze(dim=-1).sigmoid()

            overall_predict = np.concatenate([overall_predict, negative_probabilities_swap.cpu().detach().numpy(), negative_probabilities_random_recevier.cpu().detach().numpy(), negative_probabilities_random_sender.cpu().detach().numpy()])
            overall_label = np.concatenate([overall_label, np.zeros(len(negative_probabilities_swap)), np.zeros(len(negative_probabilities_random_recevier)), np.zeros(len(negative_probabilities_random_sender))])

            labels = np.concatenate([np.ones(len(positive_probabilities_non_loop)), np.zeros(len(negative_probabilities_random_recevier))])
            predicts = torch.cat([positive_probabilities_non_loop, negative_probabilities_random_recevier], dim=0).cpu().detach().numpy()
            random_receiver_ap.append(average_precision_score(labels, predicts))
            random_receiver_auc.append(roc_auc_score(labels, predicts))

            labels = np.concatenate([np.ones(len(positive_probabilities_non_loop)), np.zeros(len(negative_probabilities_random_sender))])
            predicts = torch.cat([positive_probabilities_non_loop, negative_probabilities_random_sender], dim=0).cpu().detach().numpy()
            random_sender_ap.append(average_precision_score(labels, predicts))
            random_sender_auc.append(roc_auc_score(labels, predicts))

    overall_auc = roc_auc_score(overall_label, overall_predict)
    overall_ap = average_precision_score(overall_label, overall_predict)

    evaluate_neg_edge_sampler.negative_sample_strategy = original_sample_way

    return overall_ap, overall_auc, np.mean(hist_6h_ap), np.mean(hist_6h_auc), np.mean(hist_12h_ap), np.mean(hist_12h_auc), np.mean(hist_24h_ap), np.mean(hist_24h_auc), np.mean(random_sender_ap), np.mean(random_sender_auc), np.mean(random_receiver_ap), np.mean(random_receiver_auc), np.mean(loop_ap), np.mean(loop_auc)

