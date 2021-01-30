import argparse
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torchvision.models.resnet import resnet18
import models
from torch.autograd import Variable
from data import get_dataset
from preprocess import get_transform
from utils import *
import os
from bayes_opt import BayesianOptimization
from model_profiling import model_profiling

try:
    from apex.parallel import DistributedDataParallel as DDP
    from apex.fp16_utils import *
    from apex import amp, optimizers
    from apex.multi_tensor_apply import multi_tensor_applier
except:
    print("Install apex to use AMP")


parser = argparse.ArgumentParser(description='PyTorch ConvNet Training')

parser.add_argument('--dataset', default='cifar10', choices=['cifar10', 'cifar100', 'tiny_imagenet', 'imagenet'],
                    help='dataset')
parser.add_argument('--model', default='resnet_binary', choices=['vgg11_binary', 'resnet_binary'],
                    help='architecture')
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers (default: 0)')
parser.add_argument('--epochs', default=1000, type=int,
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    help='mini-batch size (default: 256)')
parser.add_argument('--load-pretrained', type=str2bool, default=True,
                    help='Load a pretrained model')
parser.add_argument('--download-pretrained', type=str2bool, default=False,
                    help='Load a pretrained model')
parser.add_argument('--graphs', type=str2bool, default=False,
                    help='Generate files for plotting for the paper')
parser.add_argument('--prune', type=str2bool, default=False,
                    help='Prune and re-train')
parser.add_argument('--metric', default='angle_next_layer', choices=['angle', 'distance', 'angle_next_layer', 'distance_next_layer'],
                    help='Select metric for pruning')
parser.add_argument('--network', default='BinaryNet', choices=['BinaryConnect', 'BinaryNet', 'XnorNet', 'DoReFa', 'SemiReal', 'Real'],
                    help='Select type of network')
parser.add_argument('--FLReal', type=str2bool, default=True,
                    help='First and Last layer Real')
parser.add_argument('--alpha', default=1, type=float,
                    help='alpha hyperparameter for pruning loss term')
parser.add_argument('--beta', default=0.00001, type=float,
                    help='beta hyperparameter for SO loss term (default is for imagenet)')
parser.add_argument('--amp', type=str2bool, default=False,
                    help='Automatic Mixed Precision')

# Global variables
model_exp = 0
val_loader_exp = 0
criterion_exp = 0
image_size = 0
original_n_bits = 0

def main():
    global args
    global model_exp, val_loader_exp, criterion_exp, img_size, original_n_bits
    best_prec1 = 0
    start_epoch = 0
    args = parser.parse_args()
    img_size = {'cifar10': 32, 'cifar100': 32, 'tiny_imagenet': 64, 'imagenet': 224}
    cudnn.benchmark = True

    # Save path
    if not args.FLReal:
        save_filename = './pretrained/model_' + args.dataset + '_' + args.model + '_' + args.network + '_NoFLReal.pt'
    else:
        save_filename = './pretrained/model_' + args.dataset + '_' + args.model + '_' + args.network + '.pt'

    #region *************** MODEL RELATED  ****************

    # Create model
    model_config = {'dataset': args.dataset, 'depth': 18, 'FLReal': args.FLReal}
    model = models.__dict__[args.model]
    model = model(**model_config)
    model.cuda()

    # Regime for training
    regime = getattr(model, 'regime')

    # Define an arbitrary optimizer and then replace with the optimizer define in regime
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    optimizer = adjust_optimizer(optimizer, start_epoch, regime)

    # Define loss function (criterion) and optimizer
    criterion = getattr(model, 'criterion', nn.CrossEntropyLoss)().cuda()

    # Load pretrained
    if args.load_pretrained and os.path.isfile(save_filename):
        load = torch.load(save_filename)

        # When it was trained with DataParallel, it includes 'module.'
        # Don't take the module when saving, it's better to remove it when loading,
        # because other people or the online ones might not do it.
        for k in load['model'].keys():
            load['model'][k.replace('module.', '')] = load['model'].pop(k)

        model.load_state_dict(load['model'])
        model.eval()
        optimizer.load_state_dict(load['optimizer'])
        start_epoch = load['epoch']
        best_prec1 = load['best_prec1']
        print('Loaded checkpoint {} at epoch {}, best precision: {}.'.format(
            save_filename, start_epoch, best_prec1))
    if args.download_pretrained:
        # Download pretrained
        resnet18_pretrained = resnet18(True)
        model.load_state_dict(resnet18_pretrained.state_dict(), strict=False)
        model.eval()
        start_epoch = 40
        print('Downloaded pretrained resnet18 at epoch {}, best precision: {}.'.format(
            start_epoch, best_prec1))

    # NOTE: Make sure layers are declared in order in the model
    layers = layers_list(model)
    batchnorms = layers_list_bn(model)

    # Select BinaryConnect, BinaryNet, XnorNet, SemiReal or Real
    set_layers(layers, [], 'network', args.network)

    num_parameters = sum([l.nelement() for l in model.parameters()])

    if args.amp:
        model.cuda()
        model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    else:
        # Sent to all GPUS (It'll use the gpus specified by CUDA_VISIBLE_DEVICES env var)
        model = nn.DataParallel(model).cuda()
        optimizer = adjust_optimizer(optimizer, start_epoch, regime)    # Create Cuda optimizer

    #endregion

    #region **************** DATA RELATED  ****************
    # Data loading code
    transform = {
        'train': get_transform(args.dataset, input_size=None, augment=True),
        'eval': get_transform(args.dataset, input_size=None, augment=False)
    }

    val_data = get_dataset(args.dataset, 'val', transform['eval'])
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    train_data = get_dataset(args.dataset, 'train', transform['train'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    #endregion

    #region ****************** GRAPHS ********************
    if args.graphs:

        if True:
            #region ****** GRAPHS FOR KERNEL PRUNING ******
            pruning_acc = torch.zeros(len(layers)-1, 100)
            pruning_loss = torch.zeros(len(layers)-1, 100)

            # Loss without pruning
            validate(val_loader, model, criterion, 0)

            for metric in ['angles_kernel', 'distances_kernel']:
                for ranking in ['small', 'random', 'large']:
                    print('- Metric: {}\t Ranking: {}'.format(metric, ranking))
                    for i, layer in enumerate(layers):
                        # If pruneable layer
                        if len(layers) > (i + 1):
                            # Activate kernel pruning
                            if hasattr(layer, 'kernel_pruning'):
                                layer.kernel_pruning = True
                            else:
                                continue

                            for ratio in range(0, 100, 10):
                                # Reset layers
                                reset_layers(layers, batchnorms)

                                # Get distance between vectors
                                distance = get_distance(layers, i, metric).view(-1)

                                # Rank filters
                                if ranking == 'small':
                                    _, indices = torch.sort(distance, descending=True)
                                elif ranking == 'random':
                                    _, indices = torch.sort(torch.rand(distance.size()), descending=True)
                                elif ranking == 'large':
                                    _, indices = torch.sort(distance, descending=False)

                                indices = indices[0: int(round(len(indices)*(ratio/100.0)))]

                                # Prune layer (prune is 1d already)
                                layer.pruned[indices] = 1

                                pruning_loss[i, ratio], pruning_acc[i, ratio], _ = validate(
                                    val_loader, model, criterion, ratio)

                            # De-activate kernel pruning
                            layer.kernel_pruning = False

                    # Store results
                    path = './results/paper/kernel/' + metric + '/validation/' + args.dataset + '/' + args.model + '/' + ranking
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(pruning_acc, path + '/Pruning_Accuracy.pt')
                    torch.save(pruning_loss, path + '/Pruning_Loss.pt')

            #endregion

            #region ****** GRAPHS FOR CHANNEL PRUNING ******
            pruning_acc = torch.zeros(len(layers)-1, 100)
            pruning_loss = torch.zeros(len(layers)-1, 100)

            set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
            assert layers[0].prune_or_zeroOut == 'prune', \
                    "Mode 'zeroOut' doesn't work when unpruning channels"

            # Loss without pruning
            validate(val_loader, model, criterion, 0)

            # Losses with pruning
            for metric in ['angle', 'distance', 'angle_next_layer', 'distance_next_layer']:
                for ranking in ['small', 'random', 'large']:
                    print('- Metric: {}\t Ranking: {}'.format(metric, ranking))
                    for i, layer in enumerate(layers):
                        # If pruneable layer
                        if len(layers) > (i + 1):
                            for ratio in range(0, 100, 10):
                                # Reset layers
                                reset_layers(layers, batchnorms)

                                # Get distance between vectors
                                distance = get_distance(layers, i, metric)

                                # Rank filters
                                if ranking == 'small':
                                    _, indices = torch.sort(distance, descending=True)
                                elif ranking == 'random':
                                    _, indices = torch.sort(torch.rand(distance.size()), descending=True)
                                elif ranking == 'large':
                                    _, indices = torch.sort(distance, descending=False)

                                indices = indices[0: int(round(len(indices)*(ratio/100.0)))]

                                # Prune layer
                                layer.pruned_output[indices] = 1
                                layers[i + 1].pruned_input[indices] = 1
                                batchnorms[i].pruned[indices] = 1

                                pruning_loss[i, ratio], pruning_acc[i, ratio], _ = validate(
                                    val_loader, model, criterion, ratio)

                    # Store results
                    path = './results/paper/channel/' + metric + '/validation/' + args.dataset + '/' + args.model + '/' + ranking
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(pruning_acc, path + '/Pruning_Accuracy.pt')
                    torch.save(pruning_loss, path + '/Pruning_Loss.pt')

            #endregion

        # These graphs are not representative because they don't
        # contemplate the change by pruning the previous layer
        if False:
            #region ****** GRAPHS FOR KERNEL PRUNING AVERAGE ******
            pruning_acc = torch.zeros(50)
            pruning_loss = torch.zeros(50)

            # Loss without pruning
            validate(val_loader, model, criterion, 0)

            for metric in ['angles_kernel', 'distances_kernel']:
                for ranking in ['small', 'random', 'large']:
                    print('- Metric: {}\t Ranking: {}'.format(metric, ranking))
                    # Reset layers
                    reset_layers(layers, batchnorms)
                    for ratio in range(0, 50):
                        for i, layer in enumerate(layers):
                            # If pruneable layer
                            if len(layers) > (i + 1):
                                # Activate kernel pruning
                                layer.kernel_pruning = True

                                # Get distance between vectors
                                distance = get_distance(layers, i, metric).view(-1)

                                # Rank filters
                                if ranking == 'small':
                                    _, indices = torch.sort(distance, descending=True)
                                elif ranking == 'random':
                                    _, indices = torch.sort(torch.rand(distance.size()), descending=True)
                                elif ranking == 'large':
                                    _, indices = torch.sort(distance, descending=False)

                                indices = indices[0: int(round(len(indices)*(ratio/100.0)))]

                                # Prune layer (prune is 1d already)
                                layer.pruned[indices] = 1

                        pruning_loss[ratio], pruning_acc[ratio], _ = validate(
                            val_loader, model, criterion, ratio)

                    # Store results
                    path = './results/paper/kernel_average/' + metric + '/validation/' + args.dataset + '/' + args.model + '/' + ranking
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(pruning_acc, path + '/Pruning_Accuracy.pt')
                    torch.save(pruning_loss, path + '/Pruning_Loss.pt')

            #endregion

            #region ****** GRAPHS FOR CHANNEL PRUNING AVERAGE ******
            pruning_acc = torch.zeros(50)
            pruning_loss = torch.zeros(50)

            set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
            assert layers[0].prune_or_zeroOut == 'prune', \
                    "Mode 'zeroOut' doesn't work when unpruning channels"

            # Loss without pruning
            validate(val_loader, model, criterion, 0)

            # Losses with pruning
            for metric in ['angle', 'distance', 'angle_next_layer', 'distance_next_layer']:
                for ranking in ['small', 'random', 'large']:
                    print('- Metric: {}\t Ranking: {}'.format(metric, ranking))
                    for ratio in range(0, 50):
                        # Reset layers
                        reset_layers(layers, batchnorms)
                        for i, layer in enumerate(layers):
                            # If pruneable layer
                            if len(layers) > (i + 1):
                                # Get distance between vectors
                                distance = get_distance(layers, i, metric)

                                # Rank filters
                                if ranking == 'small':
                                    _, indices = torch.sort(distance, descending=True)
                                elif ranking == 'random':
                                    _, indices = torch.sort(torch.rand(distance.size()), descending=True)
                                elif ranking == 'large':
                                    _, indices = torch.sort(distance, descending=False)

                                indices = indices[0: int(round(len(indices)*(ratio/100.0)))]

                                # Prune layer
                                layer.pruned_output[indices] = 1
                                layers[i + 1].pruned_input[indices] = 1
                                batchnorms[i].pruned[indices] = 1

                        pruning_loss[ratio], pruning_acc[ratio], _ = validate(
                            val_loader, model, criterion, ratio)

                    # Store results
                    path = './results/paper/channel_average/' + metric + '/validation/' + args.dataset + '/' + args.model + '/' + ranking
                    if not os.path.exists(path):
                        os.makedirs(path)
                    torch.save(pruning_acc, path + '/Pruning_Accuracy.pt')
                    torch.save(pruning_loss, path + '/Pruning_Loss.pt')

            #endregion

    #endregion

    #region **************** EXPERIMENT 2 - PRUNING ALL LAYERS AT THE SAME TIME *****************
    if False:
        model_exp = model
        val_loader_exp = val_loader
        criterion_exp = criterion
        max_pruning = 30

        # Loss without pruning
        validate(val_loader, model, criterion, 0)

        # Bounded region of layers parameter space
        pbounds = {}
        for i, _ in enumerate(layers):
            pbounds[str(i)] = (0, 30)

        bayesian_optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            random_state=1,
        )

        bayesian_optimizer.maximize(
            init_points=50,
            n_iter=50,
        )

        print(bayesian_optimizer.max)

    #endregion

    #region **************** EXPERIMENT 3 - PRUNING LAYER-WISE *****************
    if args.prune:
        best_prec1 = 0
        model_exp = model
        val_loader_exp = val_loader
        criterion_exp = criterion
        max_pruning = 40

        # Reset layers
        reset_layers(layers, batchnorms)

        set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
        assert layers[0].prune_or_zeroOut == 'prune', "Use mode 'prune' for profiling"

        # Get initial performance metrics
        n_macs, n_params, original_n_bits, n_secs = model_profiling(model.module, img_size[args.dataset],
                                                                    img_size[args.dataset], verbose_minimal=True)

        for i in range(len(layers)-1, -1, -1):
            set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
            assert layers[0].prune_or_zeroOut == 'prune', \
                "Mode 'zeroOut' doesn't work when unpruning channels"

            # If prunable layer
            if len(layers) > (i + 1):
                # Bounded region of layers parameter space
                pbounds = {'layer_i': (i, i), 'ratio': (0, max_pruning)}

                bayesian_optimizer = BayesianOptimization(
                    f=black_box_function_2,
                    pbounds=pbounds,
                    random_state=1,
                )

                # Suggest checking the 3 points
                bayesian_optimizer.probe(params=[i, 0], lazy=True)
                bayesian_optimizer.probe(params=[i, max_pruning/2.0], lazy=True)
                bayesian_optimizer.probe(params=[i, max_pruning], lazy=True)

                bayesian_optimizer.maximize(
                    init_points=5,
                    n_iter=5,
                )

                ratio = bayesian_optimizer.max['params']['ratio']

                # Get metric between vectors
                distance = get_distance(layers, i, args.metric)

                # Rank filters
                _, indices = torch.sort(distance, descending=True)
                indices = indices[0: int(round(len(indices) * (ratio/100.0)))]

                # Reset layer
                layers[i].pruned_output[:] = 0
                layers[i + 1].pruned_input[:] = 0
                batchnorms[i].pruned[:] = 0

                # Prune layer
                layers[i].pruned_output[indices] = 1
                layers[i + 1].pruned_input[indices] = 1
                batchnorms[i].pruned[indices] = 1

                #region ** RE-TRAIN **
                if ratio < 5:
                    continue

                set_layers(layers, batchnorms, 'prune_or_zeroOut', 'zeroOut')

                # Regime for re-training
                regime = getattr(model.module, 'ReTrain')

                # Re-train for 5 epochs to allow the network to adjust to the new channels
                for epoch in range(0, 5):
                    adjust_optimizer(optimizer, epoch, regime)

                    train(train_loader, model, criterion, epoch, optimizer)

                #endregion

        # region ** RE-TRAIN **
        set_layers(layers, batchnorms, 'prune_or_zeroOut', 'zeroOut')

        # Regime for re-training
        regime = getattr(model.module, 'ReTrain')

        # Re-train for 5 epochs
        for epoch in range(0, 10):
            adjust_optimizer(optimizer, epoch, regime)

            train(train_loader, model, criterion, epoch, optimizer)

            # Evaluate on validation
            val_loss, val_prec1, _ = validate(
                val_loader, model, criterion, epoch)

            if val_prec1 > best_prec1:
                best_prec1 = val_prec1

        # endregion

        # Sanity check: compare validation with mode 'prune' to make sure it's correct
        set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
        validate(val_loader, model, criterion, epoch)

        # Total pruned percentage
        set_layers(layers, batchnorms, 'prune_or_zeroOut', 'prune')
        assert layers[0].prune_or_zeroOut == 'prune', "Use mode 'prune' for profiling"
        n_macs, n_params, n_bits, n_secs = model_profiling(model.module, img_size[args.dataset],
                                                                         img_size[args.dataset], verbose_minimal=True)
        n_params_original = sum([l.nelement() for l in model.parameters()])
        net_percent_params = (n_params * 100.0) / n_params_original
        net_percent_bits = (n_bits * 100.0) / original_n_bits

        print('\nModel: {}\t Network: {}\t Dataset: {}\t Metric: {}'.format(args.model, args.network,
                                                                            args.dataset, args.metric))
        print('Final net parameters(%): {}\t net size(%): {} \nAccuracy: {} \t N_params: {}\t '
              'Size(bits): {}\t nanosec: {}'.format(net_percent_params, net_percent_bits,
                                                    best_prec1, n_params, n_bits, n_secs))

    #endregion

    #region ************** REGULAR TRAINING ***************
    if (not args.graphs) and (not args.prune):
        for epoch in range(start_epoch, args.epochs):
            adjust_optimizer(optimizer, epoch, regime)

            # Reset layers
            reset_layers(layers, batchnorms)

            # Set pruning mode to zero out ('prune' mode doesn't work for training, only evaluation,
            # because of batch norm running mean and variance issue)
            set_layers(layers, batchnorms, 'prune_or_zeroOut', 'zeroOut')
            assert layers[0].prune_or_zeroOut == 'zeroOut', \
                "Mode 'prune' doesn't work during training because of batchnorm statistics issue"

            # Train for one epoch
            train_loss, train_prec1, _ = train(
                train_loader, model, criterion, epoch, optimizer)

            # Evaluate on validation set using binary version
            val_loss, val_prec1, _ = validate(
                val_loader, model, criterion, epoch)

            # Save checkpoint
            if val_prec1 > best_prec1:
                best_prec1 = val_prec1

                for p in list(model.parameters()):
                    if hasattr(p, 'org'):
                        p.data.copy_(p.org)

                torch.save(
                    {
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch,
                        'best_prec1': best_prec1,
                    },
                    save_filename)

    #endregion


def forward(data_loader, model, criterion, epoch=0, training=True, optimizer=None):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    SO_losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()

    # To avoid storing gradients for the inputs during evaluation
    if training:
        torch.set_grad_enabled(True)
    else:
        torch.set_grad_enabled(False)

    for i, (inputs, targets) in enumerate(data_loader):
        targets = targets.cuda()
        inputs = inputs.cuda()

        inputs_var = Variable(inputs)
        targets_var = Variable(targets)

        # measure data loading time
        data_time.update(time.time() - end)

        # Compute output
        output = model(inputs_var)

        # Soft Orthogonality Regularization
        layers = layers_list(model.module)

        SO_loss = map(soft_orthogonality_loss, layers)
        SO_loss = sum(SO_loss)

        # SO_loss = 0
        # for i, layer in enumerate(layers):
        #     # If pruneable layer
        #     if len(layers) > (i + 1):
        #         W = layer.weight.view(-1, layer.weight.size(0))
        #         I = torch.eye(W.size(1)).cuda()
        #         SO_loss += torch.norm(torch.matmul(torch.t(W), W) - I, p='fro')

        loss = criterion(output, targets_var) + args.beta*SO_loss

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, targets, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        SO_losses.update(SO_loss.data.item(), inputs.size(0))
        top1.update(prec1.item(), inputs.size(0))
        top5.update(prec5.item(), inputs.size(0))

        if training:
            # Compute gradient and do SGD step
            optimizer.zero_grad()

            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.data.copy_(p.org)
                    if args.network == 'XnorNet':
                        update_binary_grad(p)

            optimizer.step()

            for p in list(model.parameters()):
                if hasattr(p, 'org'):
                    p.org.copy_(p.data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

    # Print whole set stats
    print('{phase} - Epoch: [{0}]\t'
             'Time ({batch_time.avg:.3f})\t'
             'Data ({data_time.avg:.3f})\t'
             'Loss ({loss.avg:.3f} + {SO_loss.avg:.3f})\t'
             'Prec@1 ({top1.avg:.2f})'.format(
                epoch, phase='TRAINING' if training else 'EVALUATING',
                batch_time=batch_time,
                data_time=data_time, loss=losses, SO_loss=SO_losses, top1=top1))

    return losses.avg, top1.avg, top5.avg


def train(data_loader, model, criterion, epoch, optimizer):
    # switch to train mode
    model.train()
    return forward(data_loader, model, criterion, epoch,
                   training=True, optimizer=optimizer)


def validate(data_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()
    return forward(data_loader, model, criterion, epoch,
                   training=False, optimizer=None)


# Create list with conv2d (that are not in skip connections) and linear layers only
# NOTE: Make sure layers are declared in order in the model
def layers_list(layer):
    layers = []
    for m_name in layer.__dict__['_modules']:
        m = layer.__dict__['_modules'][m_name]
        if (isinstance(m, nn.Sequential) and m_name != 'downsample') or \
                (m.__class__.__name__ == 'BasicBlock') or (m.__class__.__name__ == 'Bottleneck'):
            layers += layers_list(m)
        if (m.__class__.__name__ == 'BinarizeConv2d') or (m.__class__.__name__ == 'BinarizeLinear') \
                or (m.__class__.__name__ == 'PrunableLinear') or (m.__class__.__name__ == 'PrunableConv2d'):
            layers += [m]

    return layers


def layers_list_bn(layer):
    layers = []
    for m_name in layer.__dict__['_modules']:
        m = layer.__dict__['_modules'][m_name]
        if (isinstance(m, nn.Sequential) and m_name != 'downsample') or \
                (m.__class__.__name__ == 'BasicBlock') or (m.__class__.__name__ == 'Bottleneck'):
            layers += layers_list_bn(m)
        if (m.__class__.__name__ == 'PrunableBatchNorm2d') \
                or (m.__class__.__name__ == 'PrunableBatchNorm1d'):
            layers += [m]

    return layers


def reset_layers(layers, batchnorms):
    for l in layers:
        l.pruned_input[:] = 0
        l.pruned_output[:] = 0
        if hasattr(l, 'pruned'):
            # Used for kernel pruning
            l.pruned[:] = 0

    for b in batchnorms:
        b.pruned[:] = 0

    return


def set_layers(layers, batchnorms, attr, value):
    for l in layers:
        # Make sure not to create attributes in classes that didn't have them (e.g. PrunableConv2d with 'network' attr)
        if hasattr(l, attr):
            setattr(l, attr, value)

    for b in batchnorms:
        if hasattr(b, attr):
            setattr(b, attr, value)

    return


def black_box_function(**ratio):
    model = model_exp
    val_loader = val_loader_exp
    criterion = criterion_exp

    layers = layers_list(model)
    batchnorms = layers_list_bn(model)

    # Reset layers
    reset_layers(layers, batchnorms)

    for i, layer in enumerate(layers):
        # If prunable layer
        if len(layers) > (i + 1):
            # Get distance between vectors
            distance = get_distance(layers, i, args.metric)

            # Rank filters
            _, indices = torch.sort(distance, descending=True)
            indices = indices[0: int(round(len(indices) * (ratio[str(i)]/100.0)))]

            # Reset layer
            layers[i].pruned_output[:] = 0
            layers[i + 1].pruned_input[:] = 0
            batchnorms[i].pruned[:] = 0

            # Prune layers
            layer.pruned_output[indices] = 1
            layers[i + 1].pruned_input[indices] = 1
            batchnorms[i].pruned[indices] = 1

    loss, _, _ = validate(val_loader, model, criterion, 0)

    # INCORRECT pruning loss
    pruning_loss = 0
    for key in ratio:
        pruning_loss += ratio[key]
    pruning_loss /= len(layers)
    pruning_loss = max_pruning - pruning_loss
    beta = 0.02

    return -(loss + beta*pruning_loss)


def black_box_function_2(layer_i, ratio):
    model = model_exp
    val_loader = val_loader_exp
    criterion = criterion_exp
    layer_i = int(layer_i)

    layers = layers_list(model.module)
    batchnorms = layers_list_bn(model.module)

    # Use this method to get original number of parameters instead of the profiling method
    # because previous layers may have been pruned, and profiling won't return the
    # original number of parameters, but the current one
    n_params_original = sum([l.nelement() for l in model.parameters()])

    # Get distance between vectors
    distance = get_distance(layers, layer_i, args.metric)

    # Rank filters
    _, indices = torch.sort(distance, descending=True)
    indices = indices[0: int(round(len(indices) * (ratio/100.0)))]

    # Reset layer
    layers[layer_i].pruned_output[:] = 0
    layers[layer_i + 1].pruned_input[:] = 0
    batchnorms[layer_i].pruned[:] = 0

    # Prune layers
    layers[layer_i].pruned_output[indices] = 1
    layers[layer_i + 1].pruned_input[indices] = 1
    batchnorms[layer_i].pruned[indices] = 1

    # Add new line
    print('')

    loss, _, _ = validate(val_loader, model, criterion, float('nan'))

    n_macs, n_params, n_bits, n_secs = model_profiling(model.module, img_size[args.dataset],
                                                       img_size[args.dataset], verbose_minimal=False)

    net_percent_params = (n_params * 100.0)/n_params_original
    net_percent_bits = (n_bits * 100.0)/original_n_bits

    print('Network parameters(%): {} size(%): {}'.format(net_percent_params, net_percent_bits))

    # Loss based in number of params or number of bits
    pruning_loss = args.alpha*(net_percent_params/100) + (args.alpha/4)*(net_percent_bits/100)

    return -(loss + pruning_loss)


def get_distance(layers, i, metric):
    if metric == 'angle':
        if hasattr(layers[i], 'angles_channel'):
            distance = layers[i].angles_channel()
        else:
            distance = layers[i + 1].angles_prev_filters()
    if metric == 'distance':
        if hasattr(layers[i], 'distances_channel'):
            distance = layers[i].distances_channel()
        else:
            distance = layers[i + 1].distances_prev_filters()
    if metric == 'angle_next_layer':
        if hasattr(layers[i + 1], 'angles_prev_filters'):
            distance = layers[i + 1].angles_prev_filters()
        else:
            distance = layers[i].angles_channel()
    if metric == 'distance_next_layer':
        if hasattr(layers[i + 1], 'distances_prev_filters'):
            distance = layers[i + 1].distances_prev_filters()
        else:
            distance = layers[i].distances_channel()
    if metric == 'angles_kernel':
        distance = layers[i].angles_kernel()
    if metric == 'distances_kernel':
        distance = layers[i].distances_kernel()

    return distance


# Scale gradients for XnorNet appropiatelly
def update_binary_grad(p):
    if p.dim() != 4:
        # Only for conv layers
        return

    weight = p.data
    n = weight[0].nelement()
    s = weight.size()
    m = weight.norm(1, 3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
    m[weight.lt(-1.0)] = 0
    m[weight.gt(1.0)] = 0

    # Wrong gradient computation in paper
    # m = m.add(1.0/n).mul(1.0-1.0/s[1]).mul(n)
    # self.target_modules[index].grad.data = \
    #         self.target_modules[index].grad.data.mul(m)

    # Corrected gradient computation
    m = m.mul(p.grad.data)
    m_add = weight.sign().mul(p.grad.data)
    m_add = m_add.sum(3, keepdim=True)\
            .sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
    m_add = m_add.mul(weight.sign())
    p.grad.data = m.add(m_add).mul(1.0-1.0/s[1]).mul(n)

    return


def remove_prelu(layer):
    for m_name in layer.__dict__['_modules']:
        m = layer.__dict__['_modules'][m_name]
        if (isinstance(m, nn.Sequential) and m_name != 'downsample') or \
                (m.__class__.__name__ == 'BasicBlock') or (m.__class__.__name__ == 'Bottleneck'):
            remove_prelu(m)
        if isinstance(m, nn.PReLU):
            layer.relu = Identity()

    return


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


def soft_orthogonality_loss(layer):
    if isinstance(layer, nn.Conv2d):
        W = layer.weight.view(-1, layer.weight.size(0))
        I = torch.eye(W.size(1)).cuda()
        if layer.__class__.__name__ == 'BinarizeConv2d':
            return torch.norm(torch.matmul(torch.t(W), W)/W.size(0) - I, p='fro')
        else:
            return torch.norm(torch.matmul(torch.t(W), W) - I, p='fro')
    else:
        return 0


if __name__ == '__main__':
    main()
