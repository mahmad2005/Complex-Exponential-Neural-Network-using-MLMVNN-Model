%back propagation function to readjust each weight of each neuron of each
%layer
%The error of the kth layer is the error of (k+1) layer divided by number of neurons
%in k layer incremented by one
%Ex. Neuron(1) error = Nueron(2) error/# of neurons in layer 1

function [] = MVBackProp()

%calculate output neuron errors
neuronErrors(layers) = neuronErrors(layers) ./ (sizeofNetwork(layers-1)+1);

%looping backwards to backprop of weights
for i = numofLayers:-1:1
    %calculate the weights
    tempWeights = (1./network(i+1));
    
    %backprop weights
    %do all except initial layer
    if (i > 1)
        neuronErrors(i) = temp(2:end,:)* neuronErrors(i+1) ./ (sizeofNetwork(i-1)+1);
    else
       neuronErrors(i) = temp(2:end,:) * neuronErrors(i+1)./(inputSize+1);
    end
    
    %correct weights
    %first layers
    %w0
    network(1) = network(1)(:,1) + learningRate .* neuronErrors(1);
    %w1 to w(size)
    network(1) = network(1) + (learningRate.*neuronErros(1) * conk(inputs(begin,:));
    
    for i = 2: numofLayers
    %calculate output for layers
    %if first layer is before second layer
    if (i == 2) %calculate output = w0 + w1x1w2x2 etc
        weightedsum(1) = network(1)(:,2:end) * (inputs(begin,:)) + network(1)(:,1);
    else
        %calculate output = w0 + w1x1w2x2 etc
        weightedSum(i-1) = network(i-1)(:,2:end)*neuronOuputs(i-2)+network(i-1)(:,1);
        
     %apply activation function to get new output
end