import argparse

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    

    
    parser = argparse.ArgumentParser(description='Fake News Detection with Clustering')

    # Dataset parameters
    # weibo twitter gossipcop_origin gossipcop_glm
    parser.add_argument('--dataset', default='gossipcop_origin', type=str,
                        help='dataset name')
    parser.add_argument('--option-file', default='./options.yaml',
                        help='options file in yaml.')

    # Training parameters
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed for random, numpy and torch (default: 42).')
    # parser.add_argument('--lr', type=float, default=1e-4,
    #                     help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', type=float, default=1e-4,
                        help='weight decay (default: 1e-4)')
    # Our algorithms are trained by the Adam optimizer (Kingma & Ba, 2014) with a batch size of 128. but we now can use 32 
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size for training')

    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs to train')
    parser.add_argument('--pre-epoch', type=int, default=10,
                        help='number of pre-train epochs')
    parser.add_argument('--es-patience', type=int, default=10,
                        help='patience of early stop')

    # Model parameters
    # yj The cluster number 𝐾 are 17 and 33 for Weibo and Twitter, respectively
    # for gossipcop we set the K  is  17
    parser.add_argument('--n-clusters', type=int, default=17,
                        help='number of clusters')
    parser.add_argument('--margin', type=float, default=0.5,
                        help='the margin for unsupervised context learning')
    parser.add_argument('--margin-class', default=0.2,
                        help='the margin for context-based triplet learning')
    parser.add_argument('--lambda-cluster', type=float, default=0.2,
                        help='the weight lambda of intra-cluster loss.')
    parser.add_argument('--lambda-triplet-class', type=float, default=0.6,
                        help='the weight lambda of context-based triplet loss.')
    # ablation study
    parser.add_argument('--unspr', default=True, type=str2bool,
                        help='whether to use unsupervised context learning')
    parser.add_argument('--multicls', type=str2bool, default=True,
                        help='whether to use multi-classifier')
    parser.add_argument('--agg', type=str2bool, default=True,
                        help='whether to use aggregation')
    parser.add_argument('--avg', type=str2bool, default=False,
                        help='whether to use average')
    
    # inference
    parser.add_argument('--inference', type=str2bool, default=False,
                        help='whether to only inference')
    
    
    return parser.parse_args()
