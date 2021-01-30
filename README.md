Networks:
1. BinaryConnect: Binary Weights, Real Activations
2. BinaryNet: Binary Weights, Binary Activations
3. XnorNet: Scaled Binary Weights, Scaled Binary Activations
4. SemiReal: Real Weights, Binary Activations
5. Real: Real Weights, Real Activations

Note:
You can train a Real network, then go to the /pretrained folder, change the name to "_Real.pt" -> "_SemiReal.pt", then resume training with the semireal, then repeat steps for "BinaryConnect/BinaryNet/XnorNet".

Note:
alpha hyperpameter for pruning loss term:
"1" for VGG and Resnet with cifar10 seem to work fine.
"2" with First and Layers binary
DoReFa: 7
XnorNet: 5

FLReal:
Will turn the first layer to real weights but it DOESN'T help. Maybe there's an issue with learning rate mismatch between binary and real layers because the networks starts to learn and then suddenly the accuracy collapses after a few iterations.
The Last layer can be turned real as well, but at the moment it's hard coded to always binary (regarless of this parameter) in order to able to use the metrics of 'angle_next_layer' and 'distance_next_layer'

Pruning procedure:
It's done layer-wise bottom-up:
1. For each layer: Test accuracy and pick best trade-off between accuracy/pruning using bayesian optimization. Purple highlight indicates a better pruning ratio found by bayesian optimization for that layer
2. If layer is not pruned, then no retraining is done
2. If layer was pruned considerably, re-train for 5 epochs using re-train regime (which can be found in resnet and vgg models)
3. When all layers are pruned, re-train for another 10 epochs and store best accuracy
4. At the end the profiling numbers will be reported

Note: 
Last layers are larger, therefore more prunable, don't panic if no pruning happens in the early layers (most won't prune at all)

Note:
Resnet18 for Cifar is actually a Resnet14 1.25x wider, making it lighter but with higher accuracy.

TO-DO:
1. Implement BinaryLinear for XnortNet and DoReFa
2. Implement NiN architecture (already sort-of done)
