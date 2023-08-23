%data 
% Learn = load('Full_MNIST.txt');
% Test = load('short_MNIST.txt');

Learn = load('short_MNIST_Test.txt');
Test = load('short_MNIST_Test.txt');

Learn = load('MNIST_Test_Full.txt');
Test = load('MNIST_Test_Full.txt');

% Learn = load('seed.txt');
% Test = load('seed.txt');
 
 %intials
inputs = Learn;
discreteOutput = 1;
discreteInput = 1;
sizeOfMlmvn = [30  1];
SoftMargins = 1;
numberOfSectors = 10;
desiredAlgo = 1;  % checkig whether learning or testing, 1 means rmse calculation for learning
maxIterations = 50000;
angularGlobalThresholdValue = 0;
angularLocalThresholdValue = 0;
localThresholdValue = 0;
initialWeights = "random";


pi2 = 2*pi;
numberOfOutputs = 1; % number of output column
numberOfSectorsHalf=floor(numberOfSectors/2);
 
 
 
[rowsInputs, colsInputs] = size(inputs);
inputsPerSample = colsInputs - numberOfOutputs;
desiredOutputs = inputs(1:rowsInputs,inputsPerSample+1:end );

inputs = inputs(:,1:inputsPerSample);

if discreteOutput
	sectorSize = pi2/numberOfSectors;
end


%updating these values because of the change in size of outputs
[rowsInputs, colsInputs] = size( inputs );
%storing number of layers
numberOfLayers = length( sizeOfMlmvn );
% numberOfLayers_1 is used then in loops
numberOfLayers_1 = numberOfLayers-1;
%storing value as this variable for ease of use
numberOfInputSamples = rowsInputs;

%preallocating two arrays, which will be used for calculation the errors in
%the case the desired and actual outputs are located close to each other,
%but accross the 0/2pi border
jjj=1:numberOfOutputs;
iii=1:numberOfInputSamples;

%preallocating a matrix to hold temporary output of the network for each
%sample
networkOutputs(1:numberOfInputSamples,1:numberOfOutputs) = 0;
if SoftMargins
	networkAngleOutputs = networkOutputs;
end

%initializing the variable that will hold the global error of the 
% network for each input sample
networkErrors = networkOutputs(1:end,1);
if SoftMargins
	netAngErrors = networkErrors;
end


%initialize random weights and bias
len = length( sizeOfMlmvn );
	
%preallocating the number of cells. network ends up being a 1xN
% cell vector where N is the number of layers
%network = cell( 1, len );
%beginning creation of the layers

%creating layer 0

% ( I would just copy the following in the if part of the if else for
% each case of possible initialWeights values modifying only the 
% lines where values are created to fit the desired method
% if sum( strcmp( initialWeights, 'random' ) ) > 0
% 	
%     % Re and Im parts of random weights are in the range [-0.5,0.5]
%     
% 	%each row of the network is the weights of a single neuron and neurons
% 	%in the first layer need the number of inputs per sample + 1 weights
% 	%network{1} = rand( [ sizeOfMlmvn(1), inputsPerSample+1 ] )-0.5 + (rand( [ sizeOfMlmvn(1),inputsPerSample+1 ] )-0.5) .* 1i;
%     %Weight_1 = rand( [ sizeOfMlmvn(1), inputsPerSample+1 ] )*10000;
%     network{1} = exp(1i*2*pi.*(Weight_1)/360);
% 
% 	%creating subsequent layers of the network if necessary
% 	if len > 1
% 		for ii = 2:len
% 			%for the following layers, the number of weights used is as
% 			%seen below: number of neurons of the previous layer + 1
% 			%network{ii} = rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5 + (rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5) .* 1i;
%             %Weight_2 = rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ])*10000;
%             network{ii} = exp(1i*2*pi.*(Weight_2)/360);
% 		end
%     
% 		
% 		clear ii
% 	end
% 
% end%end if
% 
% clear len



%creating a variable to hold the outputs of all the neurons accross the 
% network. It will be a cell
%array of column vectors, because the layers aren't necessarily all the 
% same sizes and having them as appropriately sized column vectors
% will allow for optimization of the speed of the calculation of the
% outputs of the layers of the network


len = length( network );
neuronOutputs = cell( 1, len );

for ii = 1:len
	neuronOutputs{ii}(1:sizeOfMlmvn(ii),1)=0;
end

% creating a variable to hold the errors of the outputs of the neurons
% ( has same size as neuronOutputs since each output will have 
%   an associated error value )
neuronErrors = neuronOutputs;

%initializing a variable (an array) to hold the weighted sums 
% of all the neurons accross the network
weightedSum = neuronOutputs;

clear len ii

weightedSum;
% a desired discrete output equals a root of unity corresponding to the
% bisector of a desitred sector
if discreteOutput
	ComplexValuedDesiredOutputs = exp((desiredOutputs+.5)*1i*sectorSize );
	learnAo = 0;
    % AngularDesiredOutputs - arguments of the discrete desired outputs
    AngularDesiredOutputs = mod(angle(ComplexValuedDesiredOutputs), pi2);
else
    % AngularDesiredOutputs - arguments of the continuous desired outputs
    AngularDesiredOutputs = desiredOutputs;
    ComplexValuedDesiredOutputs = exp( (desiredOutputs)*1i);
end

AngularDesiredOutputs;

if discreteInput
	% converting sector values (which are integers) into 
    % corresponding complex numbers located on the unit circle 
	% argumetnts of inputs
    %theta = pi2 .* (inputs) ./ numberOfSectors;
    theta = 1 .* (inputs) ./ numberOfSectors;
	
    % Re and Im parts of inputs
	[re, im] = pol2cart( theta, ones( rowsInputs, colsInputs ) ); % rCos(theta) + rSin(theta); or exp(1i*theta)
	inputs = re + im * 1i;
	clear re im
else
    %continuous inputs: inputs are arguments of complex numbers in the
    % range [0, 2pi]
    %converting angle values into complex numbers ( assumes angles are
	%given in radians )
	%[re, im] = pol2cart( inputs, ones( rowsInputs, colsInputs ) );
	%inputs = re + im * 1i;
	%clear re im
    inputs = exp(pi2 .* (inputs) .* 1i);
end

inputs;
%preallocating these since conversion between integer sector values and
%complex numbers based on these values occur every iteration
re = neuronOutputs;
im = neuronOutputs;

% (not doing this, commented out )
% %modifying following variable to make looping easier
% sizeOfMlmvn = [inputsPerSample sizeOfMlmvn];

%--------------------------------------------------------------------------
% END OF INPUT VALIDATION AND 
% VARIABLE INITIALIZATION STAGE
%--------------------------------------------------------------------------
re;
iterations = 0;



%% TESTING ALGORITHM
%--------------------------------------------------------------------------
			
			%% NET OUTPUT CALCULATION
			%******************************
			% CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE
			% SAMPLES
            % current_phase is an matrix to store arguments of the weighted
            % sums of all output neurons
            % this is a "winning argument" for multiple output neurons
            win_ang = 3*pi/2;
			current_phase = zeros(numberOfInputSamples, numberOfOutputs);
            % if more than one output neuron, we create an array win_dist
            % to hold angular differences to determine a winner
            if numberOfOutputs > 1
               win_dist = zeros(numberOfInputSamples, numberOfOutputs);
            end
			%looping through all samples
			for aa = 1:numberOfInputSamples
				
				% *** PROCESSING FIRST LAYER ***
				ii = 1;% ( ii holds current layer )
				
                Weight_1_W = Weight_1(:,2:end)/360;
                w_a = Weight_1_W+ theta(aa,:);
                w_a_exp = exp(1i*2*pi*w_a);
                w_a_exp_sum = sum(w_a_exp,2);
                %weighted_sum_1 = sum(exp(1i*2*pi*(((Weight_1(:,2:end))/360) + ( theta(aa,:) )))')';
				neuronOutputs{ii} =  w_a_exp_sum + exp(1i*2*pi.*(Weight_1(:,1))/360);
			    %neuronOutputs{ii} = exp(1i*2*pi.* Weight_1_W) * ( exp(1i*theta(aa,:)) ).' + exp(1i*2*pi.*(Weight_1(:,1))/360);
                    
                %network{1} = exp(1i*2*pi.*(Weight_1(:,1))/360);
                %exp(1i*2*pi.*(Weight_1(:,2:end))/360)
                

                %APPLYING CONTINUOUS ACTIVATION FUNCTION
                %TO THE FIRST HIDDEN LAYER NEURONS
                %% CONTINUOUS OUTPUT CALCULATION
                %output=weighted sum/abs(weighted sum)
                neuronOutputs{ii} = neuronOutputs{ii} ./ abs(neuronOutputs{ii}); 
				
				% *** PROCESSING FOLLOWING LAYERS ***
				
				%ii holds current layer
				for ii = 2:numberOfLayers_1
					neuronOutputs{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);


					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii} = neuronOutputs{ii} ./ abs(neuronOutputs{ii}); 
					
				end %end of ii for loop
				% *** PROCESSING THE OUTPUT LAYER***
				
				%ii holds current layer (numberOfLayers is the output layer
				%index)
				
                    ii = numberOfLayers;
					%CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER
					%NEURONS 

					neuronOutputs{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);
					
                    %% NETWORK OUTPUT CALCULATION
						%applying the activation function
											
					if discreteOutput

						% --- FOR DISCRETE OUTPUT ---
                        
                        % this will be the argument of output in the range
                        % [0, 2pi]
						neuronOutputs{ii} = mod( angle( neuronOutputs{ii}(:) ), pi2 );
                        % we store arguments of the weighted sums of output
                        % neurons for the ii-th input sample
                        current_phase(aa, 1:numberOfOutputs) = neuronOutputs{ii}; 
                    
                        
                        if numberOfOutputs>1
                           
                                 for pp = 1 : numberOfOutputs

                                    win_dist(aa, pp) = abs(current_phase(aa, pp) - win_ang);

                                    if (win_dist(aa, pp) > pi)

                                        win_dist(aa, pp) = 2*pi - win_dist(aa, pp);
                                    end

                                 end
                                 
                                 % deterrmines a neuron-winner by min
                                 % distance from win_ang if it does not
                                 % exceed pi/2, otherwise output is -1
                                 [min_dist, current_labels] = min(win_dist, [], 2);
                                 % current_labels contain the index of
                                 % output neuron-winner (from 1). Thus we
                                 % subtruct 1 to get the network output
                                 current_labels = current_labels - 1;
                                 % if the angular distance for a
                                 % neuron-winner exceeds pi/2 then this
                                 % means that no single neuron recognized
                                 % anything and the network output should
                                 % be equal to -1
                                 if (min_dist(aa) > pi/2)
                                     current_labels(aa) = -1;
                                 end
                                 % finaly a network output
                                 networkOutputs = current_labels; 
                     
                            
                        else
                            % this will be the discrete output (the number of
                            % sector) for a single output neuron
                            neuronOutputs{ii} = floor (neuronOutputs{ii}./sectorSize);
                        end

                    else
                        % --- FOR CONTINUOUS OUTPUT ---
                        % continuous output of the network
                        neuronOutputs{ii} = neuronOutputs{ii} ./ abs(neuronOutputs{ii}); 
          
					end
					%% END OF OUTPUT CALCULATION

				%above loop just calculated the output of the network for a
				%sample 
				
				%after calculation of the outputs of the neurons for the
				%sample copying the network output for for the sample
				%and storing in 'networkOutputs'
                
  if (discreteOutput ~= 1) || (numberOfOutputs == 1)                
                
				networkOutputs(aa,:) = ...
					neuronOutputs{ii};
  end
				
			end% end of aa for loop
			%above loop just calculated the ouputs of the network for all
			%samples
			
			% END OF SECTION CALCULATING OUTPUTS OF THE NETWORK FOR EACH OF
			% THE SAMPLES
			%***********************************
			%% END OF NET OUTPUT CALCULATION
			
			%% ANALYSYS OF THE RESULTS - NET ERROR CALCULATION
			%**************************************************************
			%CALCULATING GLOBAL ERROR AND COMPARING VALUE TO THRESHOLD
			for aa = 1:numberOfInputSamples
				
                % calculation of the squared error for each sample
              if discreteOutput==1  % discrete outputs
                  
                  if numberOfOutputs>1
                      % if the number of output neurons > 1                      
                      errors = abs( networkOutputs-desiredOutputs );
                      
                  else  % if the is a single output neuron
                    errors = abs( networkOutputs-desiredOutputs );
                    % this indicator is needed to take care of those errors
                    % whose "formal" actual value exceeds a half of the number
                    % of sectors (thus of those, which jump over pi)
                    indicator = (errors>numberOfSectorsHalf);
                
                    if (nnz(indicator)>0)
                          i3 = (indicator==1);
                          %mask(i3) = numberOfSectors;
                          %errors(i3) = mask(i3)-errors(i3);
                    end
                  end
  
              else  % continuous outputs
                   errors = abs(mod(angle(networkOutputs), pi2)-...
                   AngularDesiredOutputs);
                  % this indicator is needed to take care of those errors
                  % whose "formal" actual jump over pi                
                  indicator = (errors>pi);
                  if (nnz(indicator)>0)
                         i3 = (indicator==1);
                         mask(i3) = pi2;
                         errors(i3) = mask(i3)-errors(i3);
                  end
                                    
                  %if (indicator(iii,jjj)==1)
                  %  errors(iii,jjj)=pi2-errors(iii,jjj);
                  %end                 
                  mask=zeros(n1,n2); % resetting of mask
        
              end
            end
            
            %% END OF NET ERRORS CALCULATION
            
            %% ERRORS EVALUATION
            
              % Absolute Errors
              if numberOfOutputs>1 % if more than 1 output neuron then 
                %We will now apply the "winner take it all" principle in the following
                %manner: the output neuron whose output is closest to pi/2 determines the
                %output class
                %win_ang = pi/2;
                
                indicator = (errors > 0);
                
              else
                  networkError = errors; % for 1 output neuron networkError = errors
                  indicator = (networkError>localThresholdValue);
              end
                           
                              AbsErrors = sum(indicator);
                              Accuracy = 100-(AbsErrors/numberOfInputSamples)*100;            
            
              % Absolute Errors
              %if numberOfOutputs>1 % if more than 1 output neuron then we determine max error per output layer
              %    networkError = max(errors(iii,:))'; % max errors over output neurons
              %else
              %    networkError = errors; % for 1 output neuron networkError = errors
              %end
              %indicator = (networkError>localThresholdValue);
              %AbsErrors = sum(indicator);
              %Accuracy = 100-(AbsErrors/numberOfInputSamples)*100;
              
              %RMSE

              if numberOfOutputs>1
                  if (discreteOutput == 1)
                      networkErrors = errors.^2;
                  else
                      networkErrors = sum(errors(iii,:)'.^2)/numberOfOutputs;  
                  end
              else
                  networkErrors = errors.^2;
              end
            
			%calculating combined mse of all samples
			mse = sum( networkErrors ) / numberOfInputSamples;	
			rmse = sqrt( mse );
            
            % Output of the testing (evaluation) results
            
            fprintf('Errors = %5d   Accuracy = %6.3f%%\n', AbsErrors, Accuracy);
            fprintf('RMSE = %9.5f\n', rmse);

			%**************************************************************
%**************************************************
	
		%% END OF TESTING ALGORITHM