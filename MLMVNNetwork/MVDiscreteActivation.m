function [r, z, output] = MVDiscreteActivation(weights, data, k)
% Discrete k-valued Activation Function

z = weights(1);
for i = 1:length(data)
    z = z + weights(i+1) * data(i);
end

% Activate
angsize = 2*pi/k;
argz = angle(z);
argz = mod(argz, 2*pi);
output = fix(argz/angsize);
%eoutput = exp(1i*output*angsize);
r = exp(1i*output*angsize);

%arg = (argz/(2*pi/k));
% for j = 0:k-1
%     if (2*pi*j/k <= argz) && (argz < 2*pi*(j+1)/k)
%         r = exp(1i*2*pi*j/k);
%         break;
%     end
% end
    
end