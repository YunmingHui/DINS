import math
import logging
import time
import sys
import argparse
import torch
import numpy as np
import pickle
from pathlib import Path
import random

from evaluation.evaluation import eval_edge_prediction
from model.tgn import TGN
from utils.utils import EarlyStopMonitor, RandEdgeSampler, get_neighbor_finder
from utils.data_processing import get_data, compute_time_statistics
from utils.focal_loss import FocalLoss, ClassWeighedFocalLoss

torch.manual_seed(0)
np.random.seed(0)

### Argument and global variables
parser = argparse.ArgumentParser('TGN self-supervised training')
parser.add_argument('-d', '--data', type=str, help='Dataset name (eg. wikipedia or reddit)',
                    default='wikipedia')
parser.add_argument('--bs', type=int, default=1000, help='Batch_size')
parser.add_argument('--prefix', type=str, default='', help='Prefix to name the checkpoints')
parser.add_argument('--n_degree', type=int, default=10, help='Number of neighbors to sample')
parser.add_argument('--n_head', type=int, default=2, help='Number of heads used in attention layer')
parser.add_argument('--n_epoch', type=int, default=50, help='Number of epochs')
parser.add_argument('--n_layer', type=int, default=1, help='Number of network layers')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
parser.add_argument('--patience', type=int, default=5, help='Patience for early stopping')
parser.add_argument('--n_runs', type=int, default=1, help='Number of runs')
parser.add_argument('--drop_out', type=float, default=0.1, help='Dropout probability')
parser.add_argument('--gpu', type=int, default=0, help='Idx for the gpu to use')
parser.add_argument('--node_dim', type=int, default=100, help='Dimensions of the node embedding')
parser.add_argument('--time_dim', type=int, default=100, help='Dimensions of the time embedding')
parser.add_argument('--backprop_every', type=int, default=1, help='Every how many batches to '
                                                                  'backprop')
parser.add_argument('--use_memory', action='store_true',
                    help='Whether to augment the model with a node memory')
parser.add_argument('--embedding_module', type=str, default="graph_attention", choices=[
  "graph_attention", "graph_sum", "identity", "time"], help='Type of embedding module')
parser.add_argument('--message_function', type=str, default="identity", choices=[
  "mlp", "identity"], help='Type of message function')
parser.add_argument('--memory_updater', type=str, default="gru", choices=[
  "gru", "rnn"], help='Type of memory updater')
parser.add_argument('--aggregator', type=str, default="last", help='Type of message '
                                                                        'aggregator')
parser.add_argument('--memory_update_at_end', action='store_true',
                    help='Whether to update memory at the end or at the start of the batch')
parser.add_argument('--message_dim', type=int, default=100, help='Dimensions of the messages')
parser.add_argument('--memory_dim', type=int, default=172, help='Dimensions of the memory for '
                                                                'each user')
parser.add_argument('--different_new_nodes', action='store_true',
                    help='Whether to use disjoint set of new nodes for train and val')
parser.add_argument('--uniform', action='store_true',
                    help='take uniform sampling from temporal neighbors')
parser.add_argument('--randomize_features', action='store_true',
                    help='Whether to randomize node features')
parser.add_argument('--use_destination_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the destination node as part of the message')
parser.add_argument('--use_source_embedding_in_message', action='store_true',
                    help='Whether to use the embedding of the source node as part of the message')
parser.add_argument('--dyrep', action='store_true',
                    help='Whether to run the dyrep model')

parser.add_argument('--valid_index', type=int, help='The index of first edge in the validation set')
parser.add_argument('--test_index', type=int, help='The index of first edge in the test set')

try:
  args = parser.parse_args()
except:
  parser.print_help()
  sys.exit(0)

BATCH_SIZE = args.bs
NUM_NEIGHBORS = args.n_degree
NUM_NEG = 1
NUM_EPOCH = args.n_epoch
NUM_HEADS = args.n_head
DROP_OUT = args.drop_out
GPU = args.gpu
DATA = args.data
NUM_LAYER = args.n_layer
LEARNING_RATE = args.lr
NODE_DIM = args.node_dim
TIME_DIM = args.time_dim
USE_MEMORY = args.use_memory
MESSAGE_DIM = args.message_dim
MEMORY_DIM = args.memory_dim

VALID_INDEX = args.valid_index
TEST_INDEX = args.test_index

Path("./saved_models/").mkdir(parents=True, exist_ok=True)
Path("./saved_checkpoints/").mkdir(parents=True, exist_ok=True)
MODEL_SAVE_PATH = f'./saved_models/{args.prefix}-{args.data}.pth'
get_checkpoint_path = lambda \
    epoch: f'./saved_checkpoints/{args.prefix}-{args.data}-{epoch}.pth'

### set up logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
Path("log/").mkdir(parents=True, exist_ok=True)
fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.WARN)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)
logger.info(args)

### Extract data for training, validation and testing
node_features, edge_features, full_data, train_data, val_data, test_data = get_data(DATA, VALID_INDEX, TEST_INDEX,
                              different_new_nodes_between_val_and_test=args.different_new_nodes, randomize_features=args.randomize_features)

# Initialize training neighbor finder to retrieve temporal graph
train_ngh_finder = get_neighbor_finder(train_data, args.uniform)

# Initialize validation and test neighbor finder to retrieve temporal graph
full_ngh_finder = get_neighbor_finder(full_data, args.uniform)

# Initialize negative samplers. Set seeds for validation and testing so negatives are the same
# across different runs
# NB: in the inductive setting, negatives are sampled only amongst other new nodes
train_rand_sampler = RandEdgeSampler(train_data.sources, train_data.destinations)
val_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=0)
test_rand_sampler = RandEdgeSampler(full_data.sources, full_data.destinations, seed=2)

# Set device
device_string = 'cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu'
device = torch.device(device_string)

# Compute time statistics
mean_time_shift_src, std_time_shift_src, mean_time_shift_dst, std_time_shift_dst = \
  compute_time_statistics(full_data.sources, full_data.destinations, full_data.timestamps)

for i in range(args.n_runs):
  results_path = "results/{}_{}.pkl".format(args.prefix, i) if i > 0 else "results/{}.pkl".format(args.prefix)
  Path("results/").mkdir(parents=True, exist_ok=True)

  # Initialize Model
  tgn = TGN(neighbor_finder=train_ngh_finder, node_features=node_features,
            edge_features=edge_features, device=device,
            n_layers=NUM_LAYER,
            n_heads=NUM_HEADS, dropout=DROP_OUT, use_memory=USE_MEMORY,
            message_dimension=MESSAGE_DIM, memory_dimension=MEMORY_DIM,
            memory_update_at_start=not args.memory_update_at_end,
            embedding_module_type=args.embedding_module,
            message_function=args.message_function,
            aggregator_type=args.aggregator,
            memory_updater_type=args.memory_updater,
            n_neighbors=NUM_NEIGHBORS,
            mean_time_shift_src=mean_time_shift_src, std_time_shift_src=std_time_shift_src,
            mean_time_shift_dst=mean_time_shift_dst, std_time_shift_dst=std_time_shift_dst,
            use_destination_embedding_in_message=args.use_destination_embedding_in_message,
            use_source_embedding_in_message=args.use_source_embedding_in_message,
            dyrep=args.dyrep)
  criterion = torch.nn.BCELoss()
  optimizer = torch.optim.Adam(tgn.parameters(), lr=LEARNING_RATE)
  tgn = tgn.to(device)

  num_instance = len(train_data.sources)
  num_batch = math.ceil(num_instance / BATCH_SIZE)

  logger.info('num of training instances: {}'.format(num_instance))
  logger.info('num of batches per epoch: {}'.format(num_batch))
  idx_list = np.arange(num_instance)

  val_aps = []
  epoch_times = []
  total_epoch_times = []
  train_losses = []

  early_stopper = EarlyStopMonitor(max_round=args.patience)

  train_node_pairs2timestamps = {}
  loop_nodes = set()
  all_nodes = set()
  for i in range(len(train_data.sources)):
    if (train_data.sources[i], train_data.destinations[i]) not in train_node_pairs2timestamps:
      train_node_pairs2timestamps[(train_data.sources[i], train_data.destinations[i])] = []
    train_node_pairs2timestamps[(train_data.sources[i], train_data.destinations[i])].append(train_data.timestamps[i])
    
    if train_data.sources[i]==train_data.destinations[i]:
      loop_nodes.add(train_data.sources[i])
    
    all_nodes.add(train_data.sources[i])
    all_nodes.add(train_data.destinations[i])
    
  non_loop_nodes = all_nodes - loop_nodes

  for epoch in range(NUM_EPOCH):
    start_epoch = time.time()
    ### Training
    appeared_nodes = set()

    # Reinitialize memory of the model at the start of each epoch
    if USE_MEMORY:
      tgn.memory.__init_memory__()

    # Train using only training graph
    tgn.set_neighbor_finder(train_ngh_finder)
    m_loss = []

    logger.info('start {} epoch'.format(epoch))
    for k in range(0, num_batch, args.backprop_every):
      loss = 0
      optimizer.zero_grad()

      # Custom loop to allow to perform backpropagation only every a certain number of batches
      for j in range(args.backprop_every):
        batch_idx = k + j

        if batch_idx >= num_batch:
          continue

        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(num_instance, start_idx + BATCH_SIZE)
        sources_batch, destinations_batch = train_data.sources[start_idx:end_idx], \
                                            train_data.destinations[start_idx:end_idx]
        
        edge_idxs_batch = train_data.edge_idxs[start_idx: end_idx]
        timestamps_batch = train_data.timestamps[start_idx:end_idx]

        batch_node_pairs = set(zip(sources_batch, destinations_batch))
        for i in range(len(sources_batch)):
          appeared_nodes.add(sources_batch[i])
          appeared_nodes.add(destinations_batch[i])

        size = len(sources_batch)
        neg_source_batch, neg_dest_batch = train_rand_sampler.sample(size)

        node_pair_batch = set(zip(sources_batch, destinations_batch))  # set of node pairs in the batch

        tgn = tgn.train()

        # Random sender and receiver
        with torch.no_grad():
          pos_label = torch.ones(size, dtype=torch.float, device=device)
          neg_label = torch.zeros(size, dtype=torch.float, device=device)
        
        pos_prob, neg_prob = tgn.compute_edge_probabilities(sources_batch, destinations_batch, neg_dest_batch,
                                                            timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)

        loss += criterion(pos_prob.squeeze(), pos_label) + criterion(neg_prob.squeeze(), neg_label)

        neg_prob = tgn.compute_edge_probabilities_no_negative(neg_source_batch, destinations_batch,
                                                              timestamps_batch, edge_idxs_batch, NUM_NEIGHBORS)
        loss += criterion(neg_prob.squeeze(), neg_label)

        # Negative loop
        apperared_no_loop = list(appeared_nodes & non_loop_nodes)
        negative_source_III = []
        negative_dest_III = []
        negative_time_III = []
        for i in range(size):
          if sources_batch[i]==destinations_batch[i]:
            negative_node = random.choice(apperared_no_loop)
            negative_dest_III.append(negative_node)
            negative_source_III.append(negative_node)
            negative_time_III.append(timestamps_batch[i])
        if len(negative_source_III) > 1:
          neg_prob = tgn.compute_edge_probabilities_no_negative(np.array(negative_source_III), 
                                                                np.array(negative_dest_III), 
                                                                np.array(negative_time_III), 
                                                                np.array(edge_idxs_batch[0:len(negative_source_III)]), 
                                                                NUM_NEIGHBORS)
          with torch.no_grad():
            neg_label = torch.zeros(len(neg_prob), dtype=torch.float, device=device)
          loss += criterion(neg_prob.squeeze(), neg_label)

        # Temporal sampling
        sources_batch_appear_after, destinations_batch_appear_after, timestamps_batch_appear_after, edge_idxs_batch_appear_after = [], [], [], []
        for edge_idx in range(end_idx + 1, len(train_data.sources)):
          if (train_data.sources[edge_idx], train_data.destinations[edge_idx]) in node_pair_batch:
            sources_batch_appear_after.append(train_data.sources[edge_idx])
            destinations_batch_appear_after.append(train_data.destinations[edge_idx])
            timestamps_batch_appear_after.append(train_data.timestamps[edge_idx])
          if len(sources_batch_appear_after) == BATCH_SIZE:
            break
        if len(sources_batch_appear_after) > 0:
          pos_prob = tgn.compute_edge_probabilities_no_negative(np.array(sources_batch_appear_after), 
                                                                np.array(destinations_batch_appear_after), 
                                                                np.array(timestamps_batch_appear_after), 
                                                                np.array(edge_idxs_batch_appear_after), 
                                                                NUM_NEIGHBORS)
          with torch.no_grad():
            pos_label = torch.ones(len(sources_batch_appear_after), dtype=torch.float, device=device)
          loss += criterion(pos_prob.squeeze(), pos_label)

        # Positive enhancement
        sources_batch_neg_temporal, destinations_batch_neg_temporal, timestamps_batch_neg_temporal, edge_idxs_batch_neg_temporal = [], [], [], []
        for i in range(size):
          range_start = timestamps_batch[i] + 1
          range_end = min(train_data.timestamps[-1], range_start + 500) 
          if (range_end - range_start) <= 10:
            continue
          interval = (range_end - range_start) // 5 
          for j in range(5):
            used_timestamps = set(train_node_pairs2timestamps[(sources_batch[i], destinations_batch[i])])
            for attempt in range(100):  
              ts = random.randint(range_start + j * interval, range_start + (j + 1) * interval)
              if ts not in used_timestamps:
                sources_batch_neg_temporal.append(sources_batch[i])
                destinations_batch_neg_temporal.append(destinations_batch[i])
                timestamps_batch_neg_temporal.append(ts)
                edge_idxs_batch_neg_temporal.append(edge_idxs_batch[i])
                break
        if len(sources_batch_neg_temporal) > 0:
          neg_prob = tgn.compute_edge_probabilities_no_negative(np.array(sources_batch_neg_temporal), 
                                                                np.array(destinations_batch_neg_temporal), 
                                                                np.array(timestamps_batch_neg_temporal), 
                                                                np.array(edge_idxs_batch_neg_temporal), 
                                                                NUM_NEIGHBORS)
          with torch.no_grad():
            neg_label = torch.zeros(len(sources_batch_neg_temporal), dtype=torch.float, device=device)
          loss += criterion(neg_prob.squeeze(), neg_label)

      loss /= args.backprop_every
      loss.backward()
      optimizer.step()
      m_loss.append(loss.item())

      # Detach memory after 'args.backprop_every' number of batches so we don't backpropagate to
      # the start of time
      if USE_MEMORY:
        tgn.memory.detach_memory()

    epoch_time = time.time() - start_epoch
    epoch_times.append(epoch_time)

    ### Validation
    # Validation uses the full graph
    tgn.set_neighbor_finder(full_ngh_finder)

    if USE_MEMORY:
      # Backup memory at the end of training, so later we can restore it and use it for the
      # validation on unseen nodes
      train_memory_backup = tgn.memory.backup_memory()
    overall_ap, overall_auc, hist_6h_ap, hist_6h_auc, hist_12h_ap, hist_12h_auc, hist_24h_ap, hist_24h_auc, random_sender_ap, random_sender_auc, random_receiver_ap, random_receiver_auc, loop_ap, loop_auc = eval_edge_prediction(model=tgn,
                                    negative_edge_sampler=val_rand_sampler,
                                    data=val_data,
                                    n_neighbors=NUM_NEIGHBORS)
    if USE_MEMORY:
      val_memory_backup = tgn.memory.backup_memory()
      # Restore memory we had at the end of training to be used when validating on new nodes.
      # Also backup memory after validation so it can be used for testing (since test edges are
      # strictly later in time than validation edges)
      tgn.memory.restore_memory(train_memory_backup)

    if USE_MEMORY:
      # Restore memory we had at the end of validation
      tgn.memory.restore_memory(val_memory_backup)

    val_aps.append(overall_ap)
    train_losses.append(np.mean(m_loss))

    # Save temporary results to disk
    pickle.dump({
      "val_aps": val_aps,
      "train_losses": train_losses,
      "epoch_times": epoch_times,
      "total_epoch_times": total_epoch_times
    }, open(results_path, "wb"))

    total_epoch_time = time.time() - start_epoch
    total_epoch_times.append(total_epoch_time)

    logger.info('epoch: {} took {:.2f}s'.format(epoch, total_epoch_time))
    logger.info('Epoch mean loss: {}'.format(np.mean(m_loss)))
    logger.info(
      'overall val auc: {}, overall val ap: {}'.format(overall_auc, overall_ap))
    logger.info(
      '6h val auc: {}, 6h val ap: {}'.format(hist_6h_auc, hist_6h_ap))
    logger.info(
      '12h val auc: {}, 12h val ap: {}'.format(hist_12h_auc, hist_12h_ap))
    logger.info(
      '24h val auc: {}, 24h val ap: {}'.format(hist_24h_auc, hist_24h_ap))
    logger.info(
      'random sender val auc: {}, random sender val ap: {}'.format(random_sender_auc, random_sender_ap))
    logger.info(
      'random receiver val auc: {}, random receiver val ap: {}'.format(random_receiver_auc, random_receiver_ap))
    logger.info(
      'loop val auc: {}, loop val ap: {}'.format(loop_auc, loop_ap))

    # Early stopping
    if early_stopper.early_stop_check(overall_ap):
      logger.info('No improvement over {} epochs, stop training'.format(early_stopper.max_round))
      logger.info(f'Loading the best model at epoch {early_stopper.best_epoch}')
      best_model_path = get_checkpoint_path(early_stopper.best_epoch)
      tgn.load_state_dict(torch.load(best_model_path))
      logger.info(f'Loaded the best model at epoch {early_stopper.best_epoch} for inference')
      tgn.eval()
      break
    else:
      torch.save(tgn.state_dict(), get_checkpoint_path(epoch))

  # Training has finished, we have loaded the best model, and we want to backup its current
  # memory (which has seen validation edges) so that it can also be used when testing on unseen
  # nodes
  if USE_MEMORY:
    val_memory_backup = tgn.memory.backup_memory()

  ### Test
  tgn.embedding_module.neighbor_finder = full_ngh_finder
  overall_ap, overall_auc, hist_6h_ap, hist_6h_auc, hist_12h_ap, hist_12h_auc, hist_24h_ap, hist_24h_auc, random_sender_ap, random_sender_auc, random_receiver_ap, random_receiver_auc, loop_ap, loop_auc = eval_edge_prediction(model=tgn,
                                  negative_edge_sampler=test_rand_sampler,
                                  data=test_data,
                                  n_neighbors=NUM_NEIGHBORS)

  if USE_MEMORY:
    tgn.memory.restore_memory(val_memory_backup)

  logger.info(
    'Test statistics -- auc: {}, ap: {}'.format(overall_auc, overall_ap))
  logger.info(
    '6h test auc: {}, 6h test ap: {}'.format(hist_6h_auc, hist_6h_ap))
  logger.info(
    '12h test auc: {}, 12h test ap: {}'.format(hist_12h_auc, hist_12h_ap))
  logger.info(
    '24h test auc: {}, 24h test ap: {}'.format(hist_24h_auc, hist_24h_ap))
  logger.info(
    'random sender test auc: {}, random sender test ap: {}'.format(random_sender_auc, random_sender_ap))
  logger.info(
    'random receiver test auc: {}, random receiver test ap: {}'.format(random_receiver_auc, random_receiver_ap))
  logger.info(
    'loop test auc: {}, loop test ap: {}'.format(loop_auc, loop_ap))
  
  # Save results for this run
  pickle.dump({
    "val_aps": val_aps,
    "test_ap": overall_ap,
    "epoch_times": epoch_times,
    "train_losses": train_losses,
    "total_epoch_times": total_epoch_times
  }, open(results_path, "wb"))

  logger.info('Saving TGN model')
  if USE_MEMORY:
    # Restore memory at the end of validation (save a model which is ready for testing)
    tgn.memory.restore_memory(val_memory_backup)
  torch.save(tgn.state_dict(), MODEL_SAVE_PATH)
  logger.info('TGN model saved')
