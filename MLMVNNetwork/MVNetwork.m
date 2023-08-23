classdef MVNetwork % MLMVN
    properties
        %outputLayer = MVNeuron.empty;
        hiddenLayers = MVNeuron.empty;
        numLayers;
        networkSize;
        layers;
        inputs;
        outputs;
        maxIterations;
        expectedOuputs;
        % The tolerance limit. This indicates the acceptable degree of error.
        rmseThres;
    end
    methods % Begin Methods
        
        % Begin Constructor(s)
        function obj = MVNetwork(inputs, outputs, layers)
           
            obj.inputs = inputs;
            obj.outputs = outputs;
            obj.numLayers = length(layers);
            obj.layers = layers;
            obj.networkSize = sum(layers);
            numInputs = size(inputs, 2); % For the first iteration below, the first layer of neurons is being generated.
            for i = 1:obj.numLayers     % These all take the initial inputs, thus are informed of the number of inputs
                layer = MVNeuron.empty; % based on the initial data. For every iteration that follows, the number of
                for j = 1:layers(i)     % neurons contained in the preceding layer is used as the number of inputs for
                    layer{j,i} = MVNeuron(numInputs); % how many inputs those neurons in that layer will receive.
                end
                obj.hiddenLayers{i} = layer; % Here we add the layers into a list, horizontally.
                numInputs = layers(i); % This is where we update the number of inputs; just before transitioning to the
            end                        % generation of the next layer.
        end
        % End Constructor(s)
        
        % Activation Function
        function layersOutputs = activateNetwork(obj, data) % MVN w/ Continuous Activation Used
            layersOutputs = [];
            for i = 1:obj.numLayers
                layerOutput = zeros(obj.layers(i), 1);
                for j = 1:obj.layers(i)
                   layerOutput(j) = obj.hiddenLayers{1,1}{j,i}.activate(data);
                end
                layersOutputs = [layersOutputs layerOutput];
                data = layersOutputs(i);
            end
        end
        % End Activation Function
        
        % Testing Function#############################################
        function [] = testingNetwork(obj, inputs, outputs)
            % adjusting the inputs and 
            obj.inputs = inputs;
            obj.expectedOutputs = outputs;
            
             % for each sample input (row)
            for j = 1:length(obj.inputs(:,1))     
                netOutput = activateNetwork(obj, obj.inputs(j,:));
            end
            % return the discreet outputs of the neuron function
            obj.outputs = [obj.outputs ; netOutput];
        end
        % end of testing loop
        %###############################################################
                
        function array = errorCorrection(obj) %training loop
            % Back Propagation Error Correction Learning Rule w/ Angular RMSE
            
            % Acknowledgement of our dataset.
            obj.inputs;
            obj.outputs;        
            
            % The Learning Rate. For our purposes this is 1.
            learningRate = 1 + 0i;
            
            % Count of how many error correcting iterations were performed.
            iterations = 0;
            
            % Count of how many iterations are allowed to be attempted.
            obj.maxIterations = 200000; % Hard-coded to 200,000.
            
            % The flag to indicate whether learning was successful or not.
            flag = false; % Initially false, because no testing has been performed.
            
            % A list of the Root Mean Square Errors. This is for analysis.
            RMSEvalues = [];
            
            while flag == false && iterations < obj.maxIterations
                iterations = iterations + 1; % Update iterations. For the initial case, this indicates that we begin.
                flag = true; % Assume we learned to classify within toleration.
                
                % At this point we ask the question, does the network
                % produce the desired output? To answer this, we need to
                % know what the output layer neurons produce from their
                % activation functions for all cases, and how far that is
                % from the desired output. To do this, we must be aware of
                % what else we might need, and what information needs to be
                % preserved.
                
                % We are going to perform a RMSE calculation. This is the
                % Angular RMSE. The reason is because we are working in the
                % normalized complex domain, and so it is the angle that
                % determines the degree of error.
                
                % RMSE: Sqrt(MSE)
                % The Mean Square error is the summation of the square
                % errors of each sample's network error
                 
                % Let's calculate delta(r), which is the squared network
                % error for the r-th learning sample. 
                
                % BEGIN: Root Mean Square Error
                % BEGIN: Mean Square Error
                MSE = 0;
                for i = 1:size(obj.inputs,1) % [i] is the number of the data sample.
                    layersOutputs = activateNetwork(obj, obj.inputs);
                    networkError = 0;
                    % BEGIN: Network Square Error
                    for j = 1:obj.layers(obj.numLayers) % [j] is the number of the output neuron in a given sample.
                        networkError = networkError + (angle(obj.outputs(i,j)) - angle(layersOutputs(j, obj.numLayers)))^2; % See (*) and (**)
                    end
                    networkError = networkError / obj.layers(obj.numLayers);
                    % END: Network Square Error
                    MSE = MSE + networkError;
                end
                MSE = MSE / size(obj.inputs,1);
                % END: Mean Square Error
                RMSE = sqrt(MSE);
                % END: Root Mean Square Error
                
                % ---------------------------------------------------------
                % (*): obj.outputs(i,j): [i] is the data sample number,
                % organized by rows, while [j] is the number of the output
                % neuron corresponding to that data sample.
                
                % (**): layersOutputs(j, obj.numLayers): [j] is the row number
                % because each layer is a different column, and so we are
                % working at the fixed position of the output layer
                % (located at obj.numLayers, aka the final layer), while
                % iterating down the row of output neuron outputs.
                % ---------------------------------------------------------                  
                
                % Calculate the Root Mean Square Error
                % Start by calculating the Mean Square Error
                mse = 0;
                for j = 1:length(inputs(:,1))
                    z = weights(1);
                    for i = 1:length(inputs(j,:))
                        z = z + weights(i+1) * inputs(j,i);
                    end
                    mse = mse + (outputs(j) - z)^2;
                end
                mse = mse / length(inputs(:,1));
                % Square root the Mean Square Error
                rmse = sqrt(mse);

                % Keep a list of all the RMSEs
                RMSEvalues = [RMSEvalues ; rmse];


                if rmse <= tolerance
                    flag = true;
                    break; % or break

                else % Begin testing each input sample
                    flag = false; % Change the flag to false, indicating failure
                    for j = 1:length(inputs(:,1)) % for each sample input (row)
                        % Calculate the weighted sum, Z
                        z = weights(1);
                        for i = 1:length(inputs(j,:))
                            z = z + weights(i+1) * inputs(j,i);
                        end

                        testOut = tanh(z); % Compute the activation value

                        if testOut ~= outputs(j)
                            delta = outputs(j) - testOut; % Determine the error difference, Desired - Actual
                            % Adjust the weights
                            weights(1) = weights(1) + learningRate * delta; % Weight 0
                            for i_ = 1:length(inputs(1,:)) % For each column / attribute
                               weights(i_+1) = weights(i_+1) + learningRate * delta * (1/inputs(j,i_));
                            end
                            % Weights adjusted
                        end % END IF STATEMENT
                    end % END FOR LOOP
                end
                % End testing each input sample
            end % End WHILE Loop

            % rmse calc
            
            
            % check condition
             if abs(angle(desiredOutput) - angle(actualOutput) ) <= tolerance
                 % You.. WIN!
                 % break the loop and begin testing
             end
             
             
             %call baclpropagation on the object
             %backpropagation(obj);
             
        end
        
    end % End Methods
end



% jalgo = [11 12 13 14 15; 
%          21 22 23 24 25; 
%          31 32 33 34 35]
% jalgo = [11 12 13 14 15; 21 22 23 24 25; 31 32 33 34 35]
%% COMPLETE
% a)The  number  of  layers  and  the  number  of  neurons  in  each  layer
% shall  be  determined  by  the  user.
%% COMPLETE (Possibly)
% b)All  hidden  neurons  in this  network  shall  have  continuous
% activation  function (hence  its  outputs  shall  be  continuous,  and
% inputs  of  neurons  from  a  following  layer  shall  be  continuous,
% accordingly),  while  output  neurons  may  have  discrete  or
% continuous  outputs  (the best is to maintain both)
%% IN-PROGRESS
% c)The network shall be able to learn a given input/output mapping using
% the backpropagation learning algorithm. Angular RMSE (or sectoral RMSE –
% in terms of the sector indexes –   for a  discrete  output)  shall  be
% used  as  a  stopping  criterion  for  the  learning  process.  The  user
% should  be  able  to  set  up  the  following  parameters  for  the
% learning  algorithm:  the  tolerance  threshold  for  RMSE,  and  the
% max  number  of  iterations  (after  reaching  this  number,  the
% learning process should stop regardless of current RMSE).
%% UNKNOWN-STATE
% d)The  trained  network  shall  be  able  to  solve  prediction  problems
% analyzing  corresponding  input data using the weights resulted from the
% learning process. This means that the network shall  be  able  to  work
% in  the  test  mode  producing  a  result  with  the  given  weights,
% without  learning.
%%