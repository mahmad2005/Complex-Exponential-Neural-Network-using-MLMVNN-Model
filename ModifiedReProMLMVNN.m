%data 
% Learn = load('short_MNIST.txt');
% Test = load('short_MNIST.txt');
Learn = load('Full_MNIST.txt');

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
maxIterations = 1000;%50000;
angularGlobalThresholdValue = 0;
angularLocalThresholdValue = 0;
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

% % ( I would just copy the following in the if part of the if else for
% % each case of possible initialWeights values modifying only the 
% % lines where values are created to fit the desired method
% if sum( strcmp( initialWeights, 'random' ) ) > 0
% 	
%     % Re and Im parts of random weights are in the range [-0.5,0.5]
%     
% 	%each row of the network is the weights of a single neuron and neurons
% 	%in the first layer need the number of inputs per sample + 1 weights
% 	%network{1} = rand( [ sizeOfMlmvn(1), inputsPerSample+1 ] )-0.5 + (rand( [ sizeOfMlmvn(1),inputsPerSample+1 ] )-0.5) .* 1i;
%     Weight_1 = rand( [ sizeOfMlmvn(1), inputsPerSample+1 ] )*10000;
%     network{1} = exp(1i*2*pi.*(Weight_1)/360);
% 
% 	%creating subsequent layers of the network if necessary
% 	if len > 1
% 		for ii = 2:len
% 			%for the following layers, the number of weights used is as
% 			%seen below: number of neurons of the previous layer + 1
% 			%network{ii} = rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5 + (rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5) .* 1i;
%             Weight_2 = rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ])*10000;
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
    theta = pi2 .* (inputs) ./ numberOfSectors;
	
    % Re and Im parts of inputs
	[re, im] = pol2cart( theta, ones( rowsInputs, colsInputs ) ); % rCos(theta) + rSin(theta); or exp(1i*theta)
	inputs = re + im * 1i;
	clear re im theta
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ANGULAR RMSE
% 
if SoftMargins && desiredAlgo == 1
    
		%% ANGULAR RMSE ALGORITHM
		finishedLearning = false;
		
		while ~finishedLearning && iterations < maxIterations
			%% CALCULATING SAMPLE OUTPUT
			iterations=iterations+1; % iterations counter
            ErrorCounter=0; % initialization of the error counter on the iteration 
			%% NET OUTPUT CALCULATION
			%******************************
			% CALCULATING THE OUTPUTS OF THE NETWORK FOR EACH OF THE
			% SAMPLES
			
			%looping through all samples
			for aa = 1:numberOfInputSamples
				
				% *** PROCESSING FIRST LAYER ***
				ii = 1;% ( ii holds current layer index)
                
                % calculating weighted sums for the 1st hidden layer
                				
				neuronOutputs{ii} = network{ii}(:,2:end) * (inputs(aa,:)).' + network{ii}(:,1);
              

					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii} = neuronOutputs{ii} ./ abs(neuronOutputs{ii}); 

					%% END OF CONTINUOUS OUTPUT CALCULATION

				
				% *** PROCESSING FOLLOWING LAYERS until the 2nd to the last***
				
				%ii holds current layer
				for ii = 2:numberOfLayers_1
					%CALCULATING WEIGHTED SUMS OF REMAINING LAYERS OF
					%NEURONS until the 2nd to the last
                    %inputs of the neurons in the layer ii are the outputs
                    %of neurons in the layer ii-1

					neuronOutputs{ii} = network{ii}(:,2:end) * neuronOutputs{ii-1} + network{ii}(:,1);
					
					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii} = neuronOutputs{ii} ./ abs(neuronOutputs{ii}); 

					%% END OF CONTINUOUS OUTPUT CALCULATION

					
				end %end of ii for loop
                
				% *** PROCESSING THE OUTPUT LAYER***
				
				%ii holds current layer (numberOfLayers is the output layer
				%index)
				    ii = numberOfLayers;
					%CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER
					%NEURONS 

					neuronOutputs{ii} = network{ii}(:,2:end) * neuronOutputs{ii-1} + network{ii}(:,1);
					
                    %% NETWORK OUTPUT CALCULATION
						%applying the activation function
                        %for angular rmse only discrete activation function
                        %is used for the output neurons
											
						% --- FOR DISCRETE OUTPUT ---
                        
                        % this will be the argument of output in the range
                        % [0, 2pi]
						neuronOutputs{ii} = mod( angle( neuronOutputs{ii}(:) ), pi2 );
                        % Actual arguments of the weighted sums of the
                        % output neurons
                        ArgActualOutputs = neuronOutputs{ii};
                        
                        % this will be the discrete output (the number of
                        % sector)
                        neuronOutputs{ii} = floor (neuronOutputs{ii}./sectorSize);


					%% END OF OUTPUT CALCULATION
                    
				%above loop just calculated the output of the network for a
				%sample 
				
				%after calculation of the outputs of the neurons for the
				%sample copying the network output for for the sample
				%and storing in 'networkOutputs'
				networkOutputs(aa,:) = neuronOutputs{ii};
				
				%% CALCULATION OF ANGULAR RMSE ERROR
						
				%calculating angular error for the aa-th learning sample
                networkAngularErrors(aa,:) = abs(AngularDesiredOutputs(aa,:)-ArgActualOutputs(:)');
                networkAngularErrors(aa,:) = mod (networkAngularErrors(aa,:),pi2);
				
                %networkAngularErrors(aa,:) = abs( bisectors(desiredOutputs(aa,:)+1) ...
				%	- mod(angle(weightedSum{ii}'),pi2) );
                % calculation of the mean angular error for the aa-th
                % learning sample over all output neurons
				netAngErrors(aa) = mean( networkAngularErrors(aa,:) );

					% calculation of the absolute error in terms of the
					% sector numbers for all output neurons
                    outputErrors = abs( networkOutputs(aa,:)-desiredOutputs(aa,:) );
                    % maximal error over all output neurons
					maxOutputError(aa)= max( outputErrors );
 
            end % end of the for aa loop over all learning samples
            %above loop just calculated the ouputs of the network for all
			%samples
            
            % Calculation of angular RMSE over all learning samples
            AngularRMSE = sqrt(sum(netAngErrors.^2)/numberOfInputSamples);
            % if AngularRMSE<globalThresholdValue, then learning has
            % finished
            check=(max(maxOutputError)==0);
            if (AngularRMSE<=angularGlobalThresholdValue) && check
                finishedLearning = true;
            end
      
            finishedLearning= finishedLearning & check;
            
            % the number of nonzero elements in maxOutputError - the number
            % of errors
            ErrorCounter=nnz(maxOutputError);
            fprintf('Iter %5d  Errors %5d  Angular RMSE %6.4f\n', iterations, ErrorCounter, AngularRMSE);
            
            ErrorCounter=0;% reset of counter of the samples required learning on the current iteration
            
			%% END OF NET ERROR CALCULATION
			
			%% LEARNING / MODIFICATION OF WEIGHTS
			% if the algorithm has not finished learning then output of the
			% network needs to be calculated again to start correction of
			% errors
            
            
			if ~finishedLearning
				
				%calculating the output of the network for each sample and
				%correcting weights if output is > localThresholdValue
				for aa = 1:numberOfInputSamples
					

                ii = 1;% ( ii holds current layer index)
                
                % calculating weighted sums for the 1st hidden layer
                				
				weightedSum{ii} = network{ii}(:,2:end) * (inputs(aa,:)).' + network{ii}(:,1);
              

					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii} = weightedSum{ii} ./ abs(weightedSum{ii}); 

					%% END OF CONTINUOUS OUTPUT CALCULATION

				
				% *** PROCESSING FOLLOWING LAYERS until the 2nd to the last***
				
				%ii holds current layer
				for ii = 2:numberOfLayers_1
					%CALCULATING WEIGHTED SUMS OF REMAINING LAYERS OF
					%NEURONS until the 2nd to the last
                    %inputs of the neurons in the layer ii are the outputs
                    %of neurons in the layer ii-1

					weightedSum{ii} = network{ii}(:,2:end) * neuronOutputs{ii-1} + network{ii}(:,1);
					
					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii} = weightedSum{ii} ./ abs(weightedSum{ii}); 

					%% END OF CONTINUOUS OUTPUT CALCULATION

					
				end %end of ii for loop
                
				% *** PROCESSING THE OUTPUT LAYER***
				
				%ii holds current layer (numberOfLayers is the output layer
				%index)
				    ii = numberOfLayers;
					%CALCULATING WEIGHTED SUMS OF THE OUTPUT LAYER
					%NEURONS 

					weightedSum{ii} = network{ii}(:,2:end) * neuronOutputs{ii-1} + network{ii}(:,1);
					
                    %% NETWORK OUTPUT CALCULATION
						%applying the activation function
                        %for angular rmse only discrete activation function
                        %is used for the output neurons
											
						% --- FOR DISCRETE OUTPUT ---
                        
                        % this will be the argument of output in the range
                        % [0, 2pi]
						neuronOutputs{ii} = mod( angle( weightedSum{ii}(:) ), pi2 );
                        % Actual arguments of the weighted sums of the
                        % output neurons
                        ArgActualOutputs = neuronOutputs{ii};                        
                        
                        % this will be the discrete output (the number of
                        % sector)
                        neuronOutputs{ii} = floor (neuronOutputs{ii}./sectorSize);
					%% END OF OUTPUT CALCULATION
					
					
				
				%we just calculated the output of the network for a
				%sample                
				
				%after calculation of the outputs of the neurons for the
				%sample copying the network output for for the sample
				%and storing in 'networkOutputs'
				networkOutputs(aa,:) = neuronOutputs{end};

					%previous loop just calculated outputs of all neurons
					%for the current sample
					
					%now checking to see if the network output for that
					%sample is <= localThresholdValue and if it isn't, then
					%correction of the weights of the network begins
										
					%% CALCULATION OF ERROR
				%calculating angular error for the aa-th learning sample
                                                                        
                networkAngularErrors(aa,:) = abs(AngularDesiredOutputs(aa,:)-ArgActualOutputs(:)');
                networkAngularErrors(aa,:) = mod (networkAngularErrors(aa,:),pi2);                

                % calculation of the mean angular error for the aa-th
                % learning sample over all output neurons
				SampleAngError = mean( networkAngularErrors(aa,:) );

					% calculation of the absolute error in terms of the
					% sector numbers for all output neurons
                    outputErrors = abs( networkOutputs(aa,:)-desiredOutputs(aa,:) );
                
                    % this indicator is needed to take care of those errors
                % whose "formal" actual value exceeds a half of the number
                % of sectors (thus of those, which jump over pi)
                indicator = (outputErrors>numberOfSectorsHalf);
                
                if (nnz(indicator)>0)
                       [i1] = find(indicator==1);
                       outputErrors(i1) = -(outputErrors(i1)-numberOfSectors);
                end                    
                    % maximal error over all output neurons
					maxOutputError= max( outputErrors );
                    % if there is a non-zero error, then it is necessary to
                    % start the actual learning process
					%% END OF CALCULATION OF ERROR
					
					%checking against the local threshold
                    check=(SampleAngError > angularLocalThresholdValue)||(maxOutputError>0);
					if check 
                        
                        ErrorCounter=ErrorCounter+1; % increment of the counter for the samples required learning on the current iteration
						
						% if greater than, then weights are corrected, else
						% nothing happens
						
						%**************************************************
						%*** NOW CALCULATING THE ERRORS OF THE NEURONS ***
						
						%calculation of errors of neurons starts at last
						%layer and moves to first layer
						
						% ** handling special case, the output layer ***
						ii = numberOfLayers;

                            % neuronOutputs will now contain normalized 
                            % weighted sums for all output neurons 
                            neuronOutputs{ii} = weightedSum{ii} ./ abs(weightedSum{ii});
                            % jj is a vector with all output neurons'
                            % indexes
 
                            % the global error for the jjj-th output neuron
                            % equals a root of unity corresponding to the
                            % desired output - normalized wightet sum for
                            % the corresponding output neuron
                            % jjj contains indexes 1:NumberOfOutputs
                            %                                                         
                            neuronErrors{ii} (jjj) = ComplexValuedDesiredOutputs(aa,jjj).' - neuronOutputs{ii}(jjj);

                % finally we obtain the output neurons' errors
				% normalizing the global errors (dividing them
				% by the (number of neurons in the preceding
				% layer+1)
				neuronErrors{ii} = neuronErrors{ii} ./ (sizeOfMlmvn(ii-1)+1);
						
						% handling the rest of the layers - ERROR
						% BACKPROPAGATION
						for ii = numberOfLayers_1:-1:1
							
                            % calculation of the reciprocal weights for the
                            % layer ii and putting them in a vector-row
							temp = ( 1./ network{ii+1} ).'; % .' is used to avoid transjugation
                            % extraction resiprocal weights corresponding
                            % only to the inputs (the 1st weight w0 will be
                            % dropped, since it is not needed for
                            % backpropagation
							temp = temp(2:end,:);
							
                            % backpropagation of the weights
							if ii > 1 % to all hidden layers except the 1st
								neuronErrors{ii} = ( temp * neuronErrors{ii+1}) ./ (sizeOfMlmvn(ii-1)+1);
                            else % to the 1st hidden layer
								neuronErrors{ii} = ( temp * neuronErrors{ii+1} ) ./ (inputsPerSample+1);
                            end % end of the if statement
                        end % end of the for lop ii over the layers
						%**************************************************
						% *** NOW CORRECTING THE WEIGHTS OF THE NETWORK ***
						
						%handling the 1st hidden layer
                            
                            % learning rate is a reciprocal absolute value
                            % of the weighted sum
							learningRate = ( 10 ./ abs( weightedSum{1} ) );
                          % all weights except bias (w0 = w(1) in Matlab)
                  
							network{1}(:,2:end) = network{1}(:,2:end) + (learningRate .* neuronErrors{1}) * conj(inputs(aa,:));
						    Weight_1(:,2:end) = angle(network{1}(:,2:end)).*(180/pi);
                            % bias (w0 = w(1) in Matlab)
							network{1}(:,1) = network{1}(:,1) + learningRate .* neuronErrors{1};
                            Weight_1(:,1) = angle(network{1}(:,1)).*(180/pi);

						%correcting following layers
						for ii = 2:numberOfLayers
							
							%**********************************************
							%calculating new output of preceding layer
                            if (ii==2) % if a preceding layer is the 1st one
                                weightedSum{1} = network{1}(:,2:end) * (inputs(aa,:)).' + network{1}(:,1);
                            else % if a preceding layer is not the 1st one
                                weightedSum{ii-1} = network{ii-1}(:,2:end) * neuronOutputs{ii-2} + network{ii-1}(:,1);
                            end
					
					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii-1} = weightedSum{ii-1} ./ abs(weightedSum{ii-1});							
						
							%**********************************************
                            % learning rate is a reciprocal absolute value
                            % of the weighted sum					
							learningRate = ( 10 ./ abs( weightedSum{ii} ) );
							
							%learningRate not used for the output layer
							%neurons
							if ii < numberOfLayers
                               %all weights except bias (w0=w(1) in Matlab)
                               
                                a1=network{ii}(:,2:end);
                                b1=neuronErrors{ii};
                                b1=learningRate .* b1;
                                c1=neuronOutputs{ii-1};
                                c1=conj(c1);
                                c1=c1.';
                                e1=a1;
                                for i1=1:sizeOfMlmvn(ii)
                                    d1=b1(i1);
                                    e1(i1,:)=d1.*c1;
                                end
                                f1=a1+e1;
                                network{ii}(:,2:end) = f1;
                                Weight_2(:,2:end) = angle(network{ii}(:,2:end)).*(180/pi);

						        % bias (w0 = w(1) in Matlab)	
								network{ii}(:,1) = network{ii}(:,1) + learningRate .* neuronErrors{ii};
                                Weight_2(:,1) = angle(network{ii}(:,1)).*(180/pi);
                            else  % correction of the output layer neurons' weights
                               %all weights except bias (w0=w(1) in Matlab)
                                
                                a1=network{ii}(:,2:end);
                                b1=neuronErrors{ii};
                                c1=neuronOutputs{ii-1};
                                c1=conj(c1);
                                c1=c1.';
                                e1=a1;
                                for i1=1:sizeOfMlmvn(ii)
                                    d1=b1(i1);
                                    e1(i1,:)=d1.*c1;
                                end
                                f1=a1+e1;
                                network{ii}(:,2:end) = f1;
                                Weight_2(:,2:end) = angle(network{ii}(:,2:end)).*(180/pi);

                                % bias (w0 = w(1) in Matlab)
								network{ii}(:,1) = network{ii}(:,1) + neuronErrors{ii};
                                Weight_2(:,1) = angle(network{ii}(:,1)).*(180/pi);
                            end % end of if ii < numberOfLayers statement
                        end %end of ii for loop over the layers
                    end % end of "if check" 
					%done with correction of weights for current sample

					%**************************************************
                    
                    network{1} = exp(1i*2*pi.*(Weight_1)/360);
                    network{2} = exp(1i*2*pi.*(Weight_2)/360);
		
				end% end of aa for loop
				
			end% end of if statement for ~finishedLearning
		
			%% END OF LEARNING
		
		end% end of '~finishedLearning' while loop	
		
		%% END OF ANGULAR RMSE ALGORITHM	
else
end 