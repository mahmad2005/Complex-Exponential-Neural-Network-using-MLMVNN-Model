function [] = MVAngularRMSE ()
%calculating the angular RMSE for all the learning samples
angularErrors(begin,:) = abs(actualOutput(:)'- desiredOutput(begin,:));

%modding the entire error over 2pi
angularErrors(begin,:) = mod(angularErrors(begin,:),2*pi);

%obtaining the mean of all the angular errors
meanAngularErrors(begin)= mean(angularErrors(begin,:));

%calculate the abs error for all output neurons
outputError = abs(actualOutput(begin,:) - desiredOutput(begin,:));

%getting the max output error out of all the output Errors for checks
maxOutputError (begin) = max(outputError);

%calculate RMSE for angular Errors
angularRMSE = sqrt(sum(meanAngularErrors.^2)/numOfSamples);

%check to see if it needs to Back Prop or no further learning is required
if (angularRMSE <= threshold && max(maxoutputError) == 0)
    learningDone = 1;
else
    %apply backprop
end

end