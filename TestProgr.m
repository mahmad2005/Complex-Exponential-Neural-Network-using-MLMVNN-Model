% Learn = load('arithm_prog_learn.txt');
% Test = load('arithm_prog_test.txt');

%  Learn = load('short_MNIST.txt');
%  Test = load('short_MNIST.txt');

Learn = load('seed.txt');
Test = load('seed.txt');

%  Learn = load('sinx.txt');
%  Test = load('sinx.txt');


Results = MLMVN('sizeOfMlmvn', [40 1], 'inputs', Learn, 'stoppingCriteria', 'rmse', 'discreteInput', 0, 'discreteOutput', 1, 'globalthresholdvalue', 0, 'localThresholdValue', 0, 'SoftMargins', 1, 'angularGlobalThresholdValue', 0.001, 'angularLocalThresholdValue', 0,'initialWeights','random', 'numberOfSectors', 4, 'maxIterations', 100);
Weights = Results.network;
Prediction = MLMVN('network', Weights, 'inputs', Test, 'stoppingCriteria', 'test', 'discreteInput', 0, 'discreteOutput', 1, 'globalthresholdvalue', 0.1, 'localThresholdValue', 0, 'numberOfSectors', 4);
disp('Desired Outputs');
%disp(Prediction.DesiredOutputs);
disp('Actual Outputs');
%disp(Prediction.NetworkOutputs);
figure(1);
hold off
plot(Prediction.DesiredOutputs, 'or');
hold on;
plot(Prediction.NetworkOutputs, '*b');
