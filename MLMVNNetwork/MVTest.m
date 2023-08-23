load wdbc_MLMVN_Learning_Testing_Data_3pi2.mat;

Learn = Learning_3pi2;
Test = Testing_3pi2;

%Learn.Properties.Writable = true;
trainInputs = Learn(:, 1:30);
trainOuputs = Learn(:,31);

%Test.Properties.Writable = true;
testInputs = Test(:,1:30);
testOutputs = Test(:,31);

network = MVNetwork(trainInputs, trainOuputs, [30 10 1]);

%training
errC = errorCorrection(network);

%testing
%test = testingNetwork(object, testInputs, testOutputs);

%error calculation
%########################
%need to implemenet error calculations###############

%#################################################

%plotting function
plotMVNNoutputs(network.outputs, network.expectedOuputs);