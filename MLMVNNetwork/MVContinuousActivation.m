function [z, output] = MVContinuousActivation(weights, data)
% Continuous-valued MVN Activation Function
z = weights(1);
for i = 1:length(data)
    z = z + weights(i+1) * data(i);
end
output = exp(1i * angle(z));
end