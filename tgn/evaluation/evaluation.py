import math

import numpy as np
import torch
import random
from sklearn.metrics import average_precision_score, roc_auc_score


def eval_edge_prediction(
    model, negative_edge_sampler, data, n_neighbors, batch_size=200
):
    # Ensures the random sampler uses a seed for evaluation (i.e. we sample always the same
    # negatives for validation / test set)
    assert negative_edge_sampler.seed is not None
    negative_edge_sampler.reset_random_state()

    (
        random_sender_ap,
        random_sender_auc,
        random_receiver_ap,
        random_receiver_auc,
    ) = (
        [],
        [],
        [],
        [],
    )
    hist_6h_ap, hist_6h_auc, hist_12h_ap, hist_12h_auc, hist_24h_ap, hist_24h_auc = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    loop_ap, loop_auc = [], []

    all_nodes = set()
    loop_nodes = set()

    for i in range(len(data.sources)):
        all_nodes.add(data.sources[i])
        all_nodes.add(data.destinations[i])
        if data.sources[i] == data.destinations[i]:
            loop_nodes.add(data.sources[i])

    with torch.no_grad():
        model = model.eval()
        # While usually the test bach size is as big as it fits in memory, here we keep it the same
        # size as the training batch size, since it allows the memory to be updated more frequently,
        # and later test batches to access information from interactions in previous test batches
        # through the memory
        TEST_BATCH_SIZE = batch_size
        num_test_instance = len(data.sources)
        num_test_batch = math.ceil(num_test_instance / TEST_BATCH_SIZE)

        overall_prob = np.array([[]])
        overall_label = np.array([])

        for k in range(num_test_batch):
            s_idx = k * TEST_BATCH_SIZE
            e_idx = min(num_test_instance, s_idx + TEST_BATCH_SIZE)

            (
                sources_batch_no_loop,
                destinations_batch_no_loop,
                timestamps_batch_no_loop,
                edge_idxs_batch_no_loop,
            ) = ([], [], [], [])
            (
                sources_batch_loop,
                destinations_batch_loop,
                timestamps_batch_loop,
                edge_idxs_batch_loop,
            ) = ([], [], [], [])
            (
                sources_batch_all,
                destinations_batch_all,
                timestamps_batch_all,
                edge_idxs_batch_all,
            ) = ([], [], [], [])

            for i in range(s_idx, e_idx):
                if data.sources[i] == data.destinations[i]:
                    sources_batch_loop.append(data.sources[i])
                    destinations_batch_loop.append(data.destinations[i])
                    timestamps_batch_loop.append(data.timestamps[i])
                    edge_idxs_batch_loop.append(data.edge_idxs[i])
                else:
                    sources_batch_no_loop.append(data.sources[i])
                    destinations_batch_no_loop.append(data.destinations[i])
                    timestamps_batch_no_loop.append(data.timestamps[i])
                    edge_idxs_batch_no_loop.append(data.edge_idxs[i])

                sources_batch_all.append(data.sources[i])
                destinations_batch_all.append(data.destinations[i])
                timestamps_batch_all.append(data.timestamps[i])
                edge_idxs_batch_all.append(data.edge_idxs[i])

            batch_size = len(sources_batch_all)
            batch_no_loop_size = len(sources_batch_no_loop)
            batch_loop_size = len(sources_batch_loop)

            sources_batch_no_loop = np.array(sources_batch_no_loop)
            destinations_batch_no_loop = np.array(destinations_batch_no_loop)
            timestamps_batch_no_loop = np.array(timestamps_batch_no_loop)
            edge_idxs_batch_no_loop = np.array(edge_idxs_batch_no_loop)

            sources_batch_loop = np.array(sources_batch_loop)
            destinations_batch_loop = np.array(destinations_batch_loop)
            timestamps_batch_loop = np.array(timestamps_batch_loop)
            edge_idxs_batch_loop = np.array(edge_idxs_batch_loop)

            sources_batch_all = np.array(sources_batch_all)
            destinations_batch_all = np.array(destinations_batch_all)
            timestamps_batch_all = np.array(timestamps_batch_all)
            edge_idxs_batch_all = np.array(edge_idxs_batch_all)

            # Predict positive edges
            _, negatives_batch = negative_edge_sampler.sample(batch_size)
            pos_prob, neg_prob = model.compute_edge_probabilities(
                sources_batch_all,
                destinations_batch_all,
                negatives_batch,
                timestamps_batch_all,
                edge_idxs_batch_all,
                n_neighbors,
            )
            if overall_prob.size == 0:
                overall_prob = pos_prob.cpu().numpy()
            else:
                overall_prob = np.concatenate([overall_prob, pos_prob.cpu().numpy()])
            overall_label = np.concatenate([overall_label, np.ones(batch_size)])

            # Predict negative edges generated by adding 6 hours to the timestamps
            neg_prob = model.compute_edge_probabilities_no_negative(
                sources_batch_all,
                destinations_batch_all,
                timestamps_batch_all + 6 * 60 / 5,
                edge_idxs_batch_all,
                n_neighbors,
            )
            true_label = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
            overall_label = np.concatenate([overall_label, np.zeros(batch_size)])
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            hist_6h_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            hist_6h_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

            # Predict negative edges generated by adding 12 hours to the timestamps
            neg_prob = model.compute_edge_probabilities_no_negative(
                sources_batch_all,
                destinations_batch_all,
                timestamps_batch_all + 12 * 60 / 5,
                edge_idxs_batch_all,
                n_neighbors,
            )
            true_label = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
            overall_label = np.concatenate([overall_label, np.zeros(batch_size)])
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            hist_12h_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            hist_12h_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

            # Predict negative edges generated by adding 24 hours to the timestamps
            neg_prob = model.compute_edge_probabilities_no_negative(
                sources_batch_all,
                destinations_batch_all,
                timestamps_batch_all + 24 * 60 / 5,
                edge_idxs_batch_all,
                n_neighbors,
            )
            true_label = np.concatenate([np.ones(batch_size), np.zeros(batch_size)])
            overall_label = np.concatenate([overall_label, np.zeros(batch_size)])
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            hist_24h_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            hist_24h_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

            # Predict nodes that do not have a loop in validation/test set
            no_loop_nodes = all_nodes - loop_nodes
            sample_nodes = random.sample(no_loop_nodes, batch_size)
            neg_prob = model.compute_edge_probabilities_no_negative(
                np.array(sample_nodes),
                np.array(sample_nodes),
                timestamps_batch_all,
                edge_idxs_batch_all,
                n_neighbors,
            )

            overall_label = np.concatenate([overall_label, np.zeros(len(sample_nodes))])
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            true_label = np.concatenate([np.ones(batch_size), np.zeros(len(neg_prob))])
            loop_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            loop_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

            # Predict negative edges generated by randomly changing receivers
            _, negative_samples = negative_edge_sampler.sample(batch_no_loop_size)
            neg_prob = model.compute_edge_probabilities_no_negative(
                sources_batch_no_loop,
                negative_samples,
                timestamps_batch_no_loop,
                edge_idxs_batch_no_loop,
                n_neighbors,
            )
            overall_label = np.concatenate(
                [overall_label, np.zeros(batch_no_loop_size)]
            )
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            true_label = np.concatenate(
                [np.ones(len(pos_prob)), np.zeros(len(neg_prob))]
            )
            random_receiver_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            random_receiver_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

            # Predict negative edges generated by randomly changing sender
            _, negative_samples = negative_edge_sampler.sample(batch_no_loop_size)
            neg_prob = model.compute_edge_probabilities_no_negative(
                negative_samples,
                destinations_batch_no_loop,
                timestamps_batch_no_loop,
                edge_idxs_batch_no_loop,
                n_neighbors,
            )
            overall_label = np.concatenate(
                [overall_label, np.zeros(batch_no_loop_size)]
            )
            overall_prob = np.concatenate([overall_prob, neg_prob.cpu().numpy()])

            true_label = np.concatenate(
                [np.ones(len(pos_prob)), np.zeros(len(neg_prob))]
            )
            random_sender_ap.append(
                average_precision_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )
            random_sender_auc.append(
                roc_auc_score(
                    true_label,
                    np.concatenate([pos_prob.cpu().numpy(), neg_prob.cpu().numpy()]),
                )
            )

    overall_auc = roc_auc_score(overall_label, overall_prob)
    overall_ap = average_precision_score(overall_label, overall_prob)

    return (
        overall_ap,
        overall_auc,
        np.mean(hist_6h_ap),
        np.mean(hist_6h_auc),
        np.mean(hist_12h_ap),
        np.mean(hist_12h_auc),
        np.mean(hist_24h_ap),
        np.mean(hist_24h_auc),
        np.mean(random_sender_ap),
        np.mean(random_sender_auc),
        np.mean(random_receiver_ap),
        np.mean(random_receiver_auc),
        np.mean(loop_ap),
        np.mean(loop_auc),
    )


def eval_node_classification(tgn, decoder, data, edge_idxs, batch_size, n_neighbors):
    pred_prob = np.zeros(len(data.sources))
    num_instance = len(data.sources)
    num_batch = math.ceil(num_instance / batch_size)

    with torch.no_grad():
        decoder.eval()
        tgn.eval()
        for k in range(num_batch):
            s_idx = k * batch_size
            e_idx = min(num_instance, s_idx + batch_size)

            sources_batch = data.sources[s_idx:e_idx]
            destinations_batch = data.destinations[s_idx:e_idx]
            timestamps_batch = data.timestamps[s_idx:e_idx]
            edge_idxs_batch = edge_idxs[s_idx:e_idx]

            source_embedding, destination_embedding, _ = (
                tgn.compute_temporal_embeddings(
                    sources_batch,
                    destinations_batch,
                    destinations_batch,
                    timestamps_batch,
                    edge_idxs_batch,
                    n_neighbors,
                )
            )
            pred_prob_batch = decoder(source_embedding).sigmoid()
            pred_prob[s_idx:e_idx] = pred_prob_batch.cpu().numpy()

    auc_roc = roc_auc_score(data.labels, pred_prob)
    return auc_roc
