import logging
import time
import os
import numpy as np
import warnings
import shutil
import torch
import torch.nn as nn
import random

from models.GraphMixer import GraphMixer
from models.DyGFormer import DyGFormer
from models.modules import MergeLayer
from utils.utils import convert_to_gpu, create_optimizer
from utils.utils import get_neighbor_sampler, NegativeEdgeSampler
from evaluate_models_utils import evaluate_model_link_prediction
from utils.DataLoader import get_idx_data_loader, get_link_prediction_data
from utils.EarlyStopping import EarlyStopping
from utils.load_configs import get_link_prediction_args

if __name__ == "__main__":

    warnings.filterwarnings("ignore")

    # get arguments
    args = get_link_prediction_args(is_evaluation=False)

    # get data for training, validation and testing
    node_raw_features, edge_raw_features, full_data, train_data, val_data, test_data = (
        get_link_prediction_data(
            dataset_name=args.dataset_name,
            val_index=args.val_index,
            test_index=args.test_index,
        )
    )

    # initialize training neighbor sampler to retrieve temporal graph
    train_neighbor_sampler = get_neighbor_sampler(
        data=train_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=0,
    )

    # initialize validation and test neighbor sampler to retrieve temporal graph
    full_neighbor_sampler = get_neighbor_sampler(
        data=full_data,
        sample_neighbor_strategy=args.sample_neighbor_strategy,
        time_scaling_factor=args.time_scaling_factor,
        seed=1,
    )

    # initialize negative samplers, set seeds for validation and testing so negatives are the same across different runs
    # in the inductive setting, negatives are sampled only amongst other new nodes
    # train negative edge sampler does not need to specify the seed, but evaluation samplers need to do so
    train_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=train_data.src_node_ids, dst_node_ids=train_data.dst_node_ids
    )
    val_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=0
    )
    test_neg_edge_sampler = NegativeEdgeSampler(
        src_node_ids=full_data.src_node_ids, dst_node_ids=full_data.dst_node_ids, seed=2
    )

    # get data loaders
    train_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(train_data.src_node_ids))),
        batch_size=args.batch_size,
        shuffle=False,
    )
    val_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(val_data.src_node_ids))),
        batch_size=200,
        shuffle=False,
    )
    test_idx_data_loader = get_idx_data_loader(
        indices_list=list(range(len(test_data.src_node_ids))),
        batch_size=200,
        shuffle=False,
    )

    for run in range(args.num_runs):
        args.save_model_name = f"proposed"

        # set up logger
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)
        os.makedirs(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/",
            exist_ok=True,
        )
        # create file handler that logs debug and higher level messages
        fh = logging.FileHandler(
            f"./logs/{args.model_name}/{args.dataset_name}/{args.save_model_name}/{str(time.time())}.log"
        )
        fh.setLevel(logging.DEBUG)
        # create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.WARNING)
        # create formatter and add it to the handlers
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        logger.info(f"Proposed sampling")

        run_start_time = time.time()
        logger.info(f"********** Run {run + 1} starts. **********")

        logger.info(f"configuration is {args}")

        # create model
        if args.model_name == "GraphMixer":
            dynamic_backbone = GraphMixer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                num_tokens=args.num_neighbors,
                num_layers=args.num_layers,
                dropout=args.dropout,
                device=args.device,
            )
        elif args.model_name == "DyGFormer":
            dynamic_backbone = DyGFormer(
                node_raw_features=node_raw_features,
                edge_raw_features=edge_raw_features,
                neighbor_sampler=train_neighbor_sampler,
                time_feat_dim=args.time_feat_dim,
                channel_embedding_dim=args.channel_embedding_dim,
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                dropout=args.dropout,
                max_input_sequence_length=args.max_input_sequence_length,
                device=args.device,
            )
        else:
            raise ValueError(f"Wrong value for model_name {args.model_name}!")
        link_predictor = MergeLayer(
            input_dim1=node_raw_features.shape[1],
            input_dim2=node_raw_features.shape[1],
            hidden_dim=node_raw_features.shape[1],
            output_dim=1,
        )
        model = nn.Sequential(dynamic_backbone, link_predictor)

        optimizer = create_optimizer(
            model=model,
            optimizer_name=args.optimizer,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        model = convert_to_gpu(model, device=args.device)

        save_model_folder = f"./saved_models/{args.model_name}/{args.dataset_name}/{args.save_model_name}/"
        shutil.rmtree(save_model_folder, ignore_errors=True)
        os.makedirs(save_model_folder, exist_ok=True)

        early_stopping = EarlyStopping(
            patience=args.patience,
            save_model_folder=save_model_folder,
            save_model_name=args.save_model_name,
            logger=logger,
            model_name=args.model_name,
        )

        loss_func = nn.BCELoss()

        train_node_pairs2timestamps = {}
        all_nodes = set()
        loop_nodes = set()
        for i in range(len(train_data.src_node_ids)):
            if (
                train_data.src_node_ids[i],
                train_data.dst_node_ids[i],
            ) not in train_node_pairs2timestamps:
                train_node_pairs2timestamps[
                    (train_data.src_node_ids[i], train_data.dst_node_ids[i])
                ] = []
            train_node_pairs2timestamps[
                (train_data.src_node_ids[i], train_data.dst_node_ids[i])
            ].append(train_data.node_interact_times[i])

            if train_data.src_node_ids[i] == train_data.dst_node_ids[i]:
                loop_nodes.add(train_data.src_node_ids[i])

            all_nodes.add(train_data.src_node_ids[i])
            all_nodes.add(train_data.dst_node_ids[i])
        non_loop_nodes = all_nodes - loop_nodes

        for epoch in range(args.num_epochs):

            model.train()
            if args.model_name in ["GraphMixer", "DyGFormer"]:
                # training, only use training graph
                model[0].set_neighbor_sampler(train_neighbor_sampler)

            appeared_nodes = set()

            # store train losses and metrics
            train_losses = []
            for batch_idx, train_data_indices in enumerate(train_idx_data_loader):
                loss = 0
                train_data_indices = train_data_indices.numpy()
                (
                    batch_src_node_ids,
                    batch_dst_node_ids,
                    batch_node_interact_times,
                    batch_edge_ids,
                ) = (
                    train_data.src_node_ids[train_data_indices],
                    train_data.dst_node_ids[train_data_indices],
                    train_data.node_interact_times[train_data_indices],
                    train_data.edge_ids[train_data_indices],
                )

                batch_node_pairs = set(zip(batch_src_node_ids, batch_dst_node_ids))
                for i in range(len(batch_src_node_ids)):
                    appeared_nodes.add(batch_src_node_ids[i])
                    appeared_nodes.add(batch_dst_node_ids[i])

                # Negative samples for random sender and receiver
                batch_neg_src_node_ids, batch_neg_dst_node_ids = (
                    train_neg_edge_sampler.sample(size=len(batch_src_node_ids))
                )

                # Negative samples for negative loop
                apperared_no_loop = list(appeared_nodes & non_loop_nodes)
                negative_source_III = []
                negative_dest_III = []
                negative_time_III = []
                for i in range(len(batch_src_node_ids)):
                    if batch_src_node_ids[i] == batch_dst_node_ids[i]:
                        negative_node = random.choice(apperared_no_loop)
                        negative_dest_III.append(negative_node)
                        negative_source_III.append(negative_node)
                        negative_time_III.append(batch_node_interact_times[i])

                # Negative samples for temporal sampling
                (
                    sources_batch_appear_after,
                    destinations_batch_appear_after,
                    timestamps_batch_appear_after,
                ) = ([], [], [])
                for edge_index in range(
                    train_data_indices[-1] + 1, len(train_data.src_node_ids)
                ):
                    if (
                        train_data.src_node_ids[edge_index],
                        train_data.dst_node_ids[edge_index],
                    ) in batch_node_pairs:
                        sources_batch_appear_after.append(
                            train_data.src_node_ids[edge_index]
                        )
                        destinations_batch_appear_after.append(
                            train_data.dst_node_ids[edge_index]
                        )
                        timestamps_batch_appear_after.append(
                            train_data.node_interact_times[edge_index]
                        )
                    if len(sources_batch_appear_after) == len(batch_src_node_ids):
                        break

                # Positive samples for positive enhancement
                (
                    sources_batch_neg_temporal,
                    destinations_batch_neg_temporal,
                    timestamps_batch_neg_temporal,
                ) = ([], [], [])
                for i in range(len(batch_src_node_ids)):
                    range_start = batch_node_interact_times[i] + 1
                    range_end = min(
                        train_data.node_interact_times[-1], range_start + 500
                    )
                    if (range_end - range_start) <= 10:
                        continue
                    interval = (range_end - range_start) // 5
                    for j in range(5):
                        used_timestamps = set(
                            train_node_pairs2timestamps[
                                (batch_src_node_ids[i], batch_dst_node_ids[i])
                            ]
                        )
                        for attempt in range(100):
                            ts = random.randint(
                                range_start + j * interval,
                                range_start + (j + 1) * interval,
                            )
                            if ts not in used_timestamps:
                                sources_batch_neg_temporal.append(batch_src_node_ids[i])
                                destinations_batch_neg_temporal.append(
                                    batch_dst_node_ids[i]
                                )
                                timestamps_batch_neg_temporal.append(ts)
                                break

                # we need to compute for positive and negative edges respectively, because the new sampling strategy (for evaluation) allows the negative source nodes to be
                # different from the source nodes, this is different from previous works that just replace destination nodes with negative destination nodes
                if args.model_name in ["GraphMixer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                        num_neighbors=args.num_neighbors,
                        time_gap=args.time_gap,
                    )

                    if len(negative_source_III) > 0:
                        # get temporal embedding of negative source and negative destination nodes for negative samples negative loop
                        (
                            batch_neg_src_node_embeddings_III,
                            batch_neg_dst_node_embeddings_III,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(negative_source_III),
                            dst_node_ids=np.array(negative_dest_III),
                            node_interact_times=np.array(negative_time_III),
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )

                    if len(sources_batch_appear_after) > 0:
                        # get temporal embedding of negative source and negative destination nodes for negative samples temporal sampling
                        (
                            batch_neg_src_node_embeddings_IV,
                            batch_neg_dst_node_embeddings_IV,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(sources_batch_appear_after),
                            dst_node_ids=np.array(destinations_batch_appear_after),
                            node_interact_times=np.array(timestamps_batch_appear_after),
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )

                    if len(sources_batch_neg_temporal) > 0:
                        # get temporal embedding of negative source and negative destination nodes for positive samples positive enhancement
                        (
                            batch_neg_src_node_embeddings_V,
                            batch_neg_dst_node_embeddings_V,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(sources_batch_neg_temporal),
                            dst_node_ids=np.array(destinations_batch_neg_temporal),
                            node_interact_times=np.array(timestamps_batch_neg_temporal),
                            num_neighbors=args.num_neighbors,
                            time_gap=args.time_gap,
                        )

                elif args.model_name in ["DyGFormer"]:
                    # get temporal embedding of source and destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    batch_src_node_embeddings, batch_dst_node_embeddings = model[
                        0
                    ].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_src_node_ids,
                        dst_node_ids=batch_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )

                    # get temporal embedding of negative source and negative destination nodes
                    # two Tensors, with shape (batch_size, node_feat_dim)
                    (
                        batch_neg_src_node_embeddings,
                        batch_neg_dst_node_embeddings,
                    ) = model[0].compute_src_dst_node_temporal_embeddings(
                        src_node_ids=batch_neg_src_node_ids,
                        dst_node_ids=batch_neg_dst_node_ids,
                        node_interact_times=batch_node_interact_times,
                    )

                    if len(negative_source_III) > 0:
                        # get temporal embedding of negative source and negative destination nodes for negative samples negative loop
                        (
                            batch_neg_src_node_embeddings_III,
                            batch_neg_dst_node_embeddings_III,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(negative_source_III),
                            dst_node_ids=np.array(negative_dest_III),
                            node_interact_times=np.array(negative_time_III),
                        )

                    if len(sources_batch_appear_after) > 0:
                        # get temporal embedding of negative source and negative destination nodes for negative samples temporal sampling
                        (
                            batch_neg_src_node_embeddings_IV,
                            batch_neg_dst_node_embeddings_IV,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(sources_batch_appear_after),
                            dst_node_ids=np.array(destinations_batch_appear_after),
                            node_interact_times=np.array(timestamps_batch_appear_after),
                        )

                    if len(sources_batch_neg_temporal) > 0:
                        # get temporal embedding of negative source and negative destination nodes for positive samples positive enhancement
                        (
                            batch_neg_src_node_embeddings_V,
                            batch_neg_dst_node_embeddings_V,
                        ) = model[0].compute_src_dst_node_temporal_embeddings(
                            src_node_ids=np.array(sources_batch_neg_temporal),
                            dst_node_ids=np.array(destinations_batch_neg_temporal),
                            node_interact_times=np.array(timestamps_batch_neg_temporal),
                        )
                else:
                    raise ValueError(f"Wrong value for model_name {args.model_name}!")
                # get positive and negative probabilities, shape (batch_size, )
                positive_probabilities = (
                    model[1](
                        input_1=batch_src_node_embeddings,
                        input_2=batch_dst_node_embeddings,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                negative_probabilities = (
                    model[1](
                        input_1=batch_src_node_embeddings,
                        input_2=batch_neg_dst_node_embeddings,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                predicts = torch.cat(
                    [positive_probabilities, negative_probabilities], dim=0
                )
                labels = torch.cat(
                    [
                        torch.ones_like(positive_probabilities),
                        torch.zeros_like(negative_probabilities),
                    ],
                    dim=0,
                )
                loss += loss_func(input=predicts, target=labels)

                negative_probabilities = (
                    model[1](
                        input_1=batch_neg_src_node_embeddings,
                        input_2=batch_dst_node_embeddings,
                    )
                    .squeeze(dim=-1)
                    .sigmoid()
                )
                labels = torch.zeros_like(negative_probabilities)
                loss += loss_func(input=negative_probabilities, target=labels)

                if len(negative_source_III) > 0:
                    negative_probabilities = (
                        model[1](
                            input_1=batch_neg_src_node_embeddings_III,
                            input_2=batch_neg_dst_node_embeddings_III,
                        )
                        .squeeze(dim=-1)
                        .sigmoid()
                    )
                    labels = torch.zeros_like(negative_probabilities)
                    loss += loss_func(input=negative_probabilities, target=labels)

                if len(sources_batch_appear_after) > 0:
                    positive_probabilities = (
                        model[1](
                            input_1=batch_neg_src_node_embeddings_IV,
                            input_2=batch_neg_dst_node_embeddings_IV,
                        )
                        .squeeze(dim=-1)
                        .sigmoid()
                    )
                    labels = torch.ones_like(positive_probabilities)
                    loss += loss_func(input=positive_probabilities, target=labels)

                if len(sources_batch_neg_temporal) > 0:
                    negative_probabilities = (
                        model[1](
                            input_1=batch_neg_src_node_embeddings_V,
                            input_2=batch_neg_dst_node_embeddings_V,
                        )
                        .squeeze(dim=-1)
                        .sigmoid()
                    )
                    labels = torch.zeros_like(negative_probabilities)
                    loss += loss_func(input=negative_probabilities, target=labels)

                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            (
                overall_ap,
                overall_auc,
                hist_6h_ap,
                hist_6h_auc,
                hist_12h_ap,
                hist_12h_auc,
                hist_24h_ap,
                hist_24h_auc,
                random_sender_ap,
                random_sender_auc,
                random_receiver_ap,
                random_receiver_auc,
                loop_ap,
                loop_auc,
            ) = evaluate_model_link_prediction(
                model_name=args.model_name,
                model=model,
                neighbor_sampler=full_neighbor_sampler,
                evaluate_idx_data_loader=val_idx_data_loader,
                evaluate_neg_edge_sampler=val_neg_edge_sampler,
                evaluate_data=val_data,
                loss_func=loss_func,
                num_neighbors=args.num_neighbors,
                time_gap=args.time_gap,
            )

            logger.info(
                f'Epoch: {epoch + 1}, learning rate: {optimizer.param_groups[0]["lr"]}, train loss: {np.mean(train_losses):.4f}'
            )
            logger.info(
                "overall val auc: {}, overall val ap: {}".format(
                    overall_auc, overall_ap
                )
            )
            logger.info("6h val auc: {}, 6h val ap: {}".format(hist_6h_auc, hist_6h_ap))
            logger.info(
                "12h val auc: {}, 12h val ap: {}".format(hist_12h_auc, hist_12h_ap)
            )
            logger.info(
                "24h val auc: {}, 24h val ap: {}".format(hist_24h_auc, hist_24h_ap)
            )
            logger.info(
                "random sender val auc: {}, random sender val ap: {}".format(
                    random_sender_auc, random_sender_ap
                )
            )
            logger.info(
                "random receiver val auc: {}, random receiver val ap: {}".format(
                    random_receiver_auc, random_receiver_ap
                )
            )
            logger.info("loop val auc: {}, loop val ap: {}".format(loop_auc, loop_ap))

            # select the best model based on all the validate metrics
            early_stop = early_stopping.step(overall_ap, model)

            if early_stop:
                break

        # load the best model
        early_stopping.load_checkpoint(model)

        # evaluate the best model
        logger.info(f"get final performance on dataset {args.dataset_name}...")

        (
            overall_ap,
            overall_auc,
            hist_6h_ap,
            hist_6h_auc,
            hist_12h_ap,
            hist_12h_auc,
            hist_24h_ap,
            hist_24h_auc,
            random_sender_ap,
            random_sender_auc,
            random_receiver_ap,
            random_receiver_auc,
            loop_ap,
            loop_auc,
        ) = evaluate_model_link_prediction(
            model_name=args.model_name,
            model=model,
            neighbor_sampler=full_neighbor_sampler,
            evaluate_idx_data_loader=test_idx_data_loader,
            evaluate_neg_edge_sampler=test_neg_edge_sampler,
            evaluate_data=test_data,
            loss_func=loss_func,
            num_neighbors=args.num_neighbors,
            time_gap=args.time_gap,
        )

        # store the evaluation metrics at the current run
        logger.info(
            "Test statistics -- auc: {}, ap: {}".format(overall_auc, overall_ap)
        )
        logger.info("6h test auc: {}, 6h test ap: {}".format(hist_6h_auc, hist_6h_ap))
        logger.info(
            "12h test auc: {}, 12h test ap: {}".format(hist_12h_auc, hist_12h_ap)
        )
        logger.info(
            "24h test auc: {}, 24h test ap: {}".format(hist_24h_auc, hist_24h_ap)
        )
        logger.info(
            "random sender test auc: {}, random sender test ap: {}".format(
                random_sender_auc, random_sender_ap
            )
        )
        logger.info(
            "random receiver test auc: {}, random receiver test ap: {}".format(
                random_receiver_auc, random_receiver_ap
            )
        )
        logger.info("loop test auc: {}, loop test ap: {}".format(loop_auc, loop_ap))

        # avoid the overlap of logs
        if run < args.num_runs - 1:
            logger.removeHandler(fh)
            logger.removeHandler(ch)
