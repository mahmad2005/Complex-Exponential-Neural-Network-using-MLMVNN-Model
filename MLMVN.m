function [output] = MLMVN(varargin)
%% VARIABLE INITIALIZATION
%*********************************************************
%-----------------------------------------------------------
%ENTERING VARIABLE DECLARATION AND INPUT VALIDATION STAGE 
%-----------------------------------------------------------
%***********************************************************

% *** DECLARING INPUT VARIABLES

%variable 1: (optional input) sizeOfMLMVN - used if an existing network is not
%passed to the function
sizeOfMlmvn = 0;%bad default value 
%variable 2: (optional input) network - the cell array of matrices of weights
network = 0;
%variable 3: inputs - the 2D matrix of inputs and desired outputs
inputs = 0;
%variable 4: stoppingCriteria - a string indicating what stopping criteria
%will be used
stoppingCriteria = 0;
%variable 5: discreteInput - either a 0 or 1 to determine if input is
%in discrete or continous form
discreteInput = -1;
%variable 6: discreteOutput - again a 0 or a 1 determine if
%output is in discrete or continous form
discreteOutput = -1;
%var 7: globalThresholdValue - a value used to calculate the stopping point
% given a stopping criteria, range = [0,inf)
globalThresholdValue = -1;
%var 8: localThresholdValue - another value used to calculate the stopping
%point of the algorithm based on the stopping criteria, range = [0,gtv]
localThresholdValue = -1;
%var 9: initialWeights - a string indicating how the weights should be
%initialize
initialWeights = -1;
%var 10: (optional) numberOfSectors - an integer value indicating the
%number of sectors used for classification in the learning algorithm. If
%discreteInput or discreteOutput then this variable is necessary
numberOfSectors = -1;
%var 11: (optional) angularGlobalThresholdValue - a floating point value
%used to determine the end of angular rmse learning
angularGlobalThresholdValue = -1;
%var 12:(optional, must be used if SoftMargins is used) - a floating point 
% value used to determine whether a current angular error is small enough
% and does not exceed this value (if it does, weights shall be adjasted
% regardless of the output correctness)
angularLocalThresholdValue = -1;
%var 13: (optional) SoftMargins - a logical or numeric 1 or 0 to indicate
%whether the angular rmse algorithm will be used
SoftMargins = -1;
%var 14: (optional) figureHandle - indicates that function was called by
%an mlmvnWindow and gives the handle of the mlmvnWindow
figureHandle = -1;

maxIterations = 5000;

%HAVEN'T DONE THIS ONE COMPLETELY BUT NECESSARY FOR CORRECTION OF WIEGHTS -
%VALUE 'learningRate' IS DESCRIBED ON PG. 5 BELOW EQUATIONS (12) AND (13)
learningRate = 0;
%FURTHER DESCRIBED AT BOTTOM OF LEFT COLUMN OF PG. 4 BELOW EQUATION (7) AND
%NECESSARY VECTOR IS:
weightedSum = 0;
%WEIGHTED SUM HOLDS THE WIEGHTED SUM "OBTAINED ON THE Rth LEARNING STEP"

%CREATING VECTOR 'bisectors' WITH THE ANGLE VALUES OF THE BISECTORS OF ALL
%SECTORS THAT WILL BE A PART OF LEARNING, INDICATED BY THE VALUE
%'numberOfSectors'
bisectors = -1;


% creating a list of stopping criterias available as
% possible checking methods for ending of the function 
acceptableStoppingCriteriaValues = ...
 { 'rmse', 'test' };

% doing the same for initial weights methods
acceptableInitialWeightsValues = ...
	{ 'random' };

%using variable 'givenInputs' to determine which values were passed to the
%function
givenInputs(1:15) = false;


% *** GATHERING INPUT FROM INPUT ARGUMENT 'varargin'
varargin = varargin.';
len = length(varargin)-1;

for c1 = 1:2:len

    if ~( ischar(varargin{c1} ) )%should be a string 
        error( ['inputs should be ordered in property name,value pairs, i.e. ' ...
            '''sizeOfMLMVN'', [3,1]' ] );
    end

    %converting string to lower case for string comparison and checking
    %cases
    switch lower( varargin{c1} )        
        case 'sizeofmlmvn'
            sizeOfMlmvn = varargin{c1+1};
            givenInputs(1) = true;
        case 'inputs'
            inputs = varargin{c1+1};
            givenInputs(2) = true;
        case 'stoppingcriteria'
            stoppingCriteria = varargin{c1+1};
            givenInputs(3) = true;
        case 'discreteinput'
            discreteInput = varargin{c1+1};
            givenInputs(4) = true;
        case 'discreteoutput'
            discreteOutput = varargin{c1+1};
            givenInputs(5) = true;
        case 'globalthresholdvalue'
            globalThresholdValue = varargin{c1+1};
            givenInputs(6) = true;
        case 'localthresholdvalue'
            localThresholdValue = varargin{c1+1};
            givenInputs(7) = true;
        case 'network'
            network = varargin{c1+1};
			givenInputs(8) = true;
		case 'initialweights'
			initialWeights = varargin{c1+1};
			givenInputs(9) = true;
		case 'numberofsectors'
			numberOfSectors = varargin{c1+1};
			givenInputs(10) = true;
		case 'angularglobalthresholdvalue'
			angularGlobalThresholdValue = varargin{c1+1};
			givenInputs(11) = true;
		case 'angularlocalthresholdvalue'
			angularLocalThresholdValue = varargin{c1+1};
			givenInputs(12) = true;            
		case 'softmargins'
			SoftMargins = varargin{c1+1};
			givenInputs(13) = true;
		case 'figurehandle'
			figureHandle = varargin{c1+1};
			givenInputs(14) = true;
        case 'maxiterations'
            maxIterations = varargin{c1+1};
            givenInputs(15) = true;
    end% of the switch

end% of for loop

clear c1

% *** VALIDATING GIVEN INPUTS

%checking givenInputs(12) first because some of the following 
% validation relies on this
if givenInputs(13)
	
	a = SoftMargins;
	
	if ~( ~iscell(a) && isscalar(a) && ( islogical(a) || isnumeric(a) ) )
		s = '\n\nSoftMargins must be scalar value, either 0 or 1\n\n';
		error( 'MLMVN:BadInput', s );
	end
	
	clear a
	
else
	SoftMargins = false;
end

% if both sizeOfMLMVN and network parameters were entered, a network cannot
% be created due to the contradiction
if (givenInputs(1)) && (givenInputs(8))
    	s = '\n\nA network shall be determined either by sizeOfMLMVN or network parameters, but not by both\n\n';
		error( 'MLMVN:BadInput', s );
end


if givenInputs(1)% if sizeOfMlmvn was passed to the function
    
    %validating sizeOfMlmvn
    len = length( size( sizeOfMlmvn ) );
    x = size( size( sizeOfMlmvn ) );
    x = x(1);
    if ( len > 2 || x ~= 1 )
        error( 'sizeOfMLMVN has wrong dimensions' );
    end

    result = sizeOfMlmvn == floor( sizeOfMlmvn );
    if  result
    else
        error( [ 'elements of sizeOfMLMVN should ' ... 
            'contain only whole numbers' ] );
    end

    if min( sizeOfMlmvn ) < 1
        error( 'values for sizeOfMLMVN should be greater than 0' );
	end
    
	%cleaning up the variables used for validation
	clear x
	clear len
	clear result
	
else% sizeOfMlmvn was not passed to the function
    
    if ~givenInputs(8)%network was not passed to the function
    %in this scenario both sizeOfMlmvn and network were not passed as
    %inputs and, for the algorithm, at least one has to be passed to the
    %function
	error( [ 'neither a size for a network nor an existing network ' ...
		'were passed to the function' ] );
	
	end
end% end of givenInputs(1) if statement

if givenInputs(2)% 'inputs' was passed
	
	%validating 'inputs'
	
	s = size( inputs );
	len = length( s );
	
	if ( len ~= 2 )
		error( '''inputs'' should be a 2d matrix of values' );
	end
	
	clear s len
else
	error( 'input matrix, ''inputs'' was not passed to the function' );
	
end

if givenInputs(3)%stoppingCriteria
	
	%validating
	
	% has to be a string
	if ischar( stoppingCriteria )
		
		%converting to lower case for convenience
		stoppingCriteria = lower( stoppingCriteria );
		
		%searching for given string in acceptable values array
		x = strcmp( acceptableStoppingCriteriaValues, stoppingCriteria );
		
		%if the string was found
		if find(x,1)
			
			%hip hip hooray, good input data
			
		else% the string was not found ( bad input )
			error( [ 'the stoppingCriteria passed to the function was invalid,' ...
				'valid values are: rmse and test' ] )
		end
		
		clear x
		
	else% stopping criteria passed to the function was not a string
		error( [ 'stoppingCriteria should be a string indicating desired' ...
			' method for algorithm ending calculation' ] );
	end
	
else% stopping criteria was not passed to the function
	error( [ 'stoppingCriteria, a necessary input value,'...
		' was not passed to the function' ] )

end%done with givenInputs(3)

if givenInputs(4)
	
	%validating 'discreteInput'
	if discreteInput == 1 || discreteInput == 0
		
		%valid input
		
	else
		
		error( ['property discreteInput should be either a 1 or a 0 ' ...
			'to indicate whether input is discrete or continous' ] )
		
	end% end of validation
	
else
	
	error( ['input was not given to indicate discrete or continuous input' ...
		' property ''discreteInput'' must be passed to the function' ] )
	
end% done with givenInputs(4)

if givenInputs(5)%discreteOutput
	
		%validating
	if isnumeric( discreteOutput ) || islogical( discreteOutput )
		
		if discreteOutput == 1 || discreteOutput == 0

			%valid input

		else

			error( ['property discreteOutput should be either a 1 or a 0 ' ...
				'to indicate whether output is discrete or continous' ] )

		end
	else
		s = [ '\n\nproperty discreteOutput should be either a 1 or a 0 ' ...
				'to indicate whether output is discrete or continous' ...
				'\n\n' ];
		error( 'MLMVN:BadInput', s );
	end% end of validation
	
 	if (SoftMargins ==1) && (discreteOutput == 0)
 			s = '\n\noutput should be discrete in the case that ';
 			s = [ s 'SoftMargins method is used \n\n'];
 			
 			error( 'MLMVN:BadInput', s );
 		
    end
    
    if (strcmp(stoppingCriteria, 'max')) && ( discreteOutput == 0)
 			s = '\n\noutput should be discrete in the case that ';
 			s = [ s 'stoppingCriteria == ''max'' \n\n'];
 			
 			error( 'MLMVN:BadInput', s );        
    end
    
else
	
	error( ['input was not given to indicate discrete or continuous output' ...
		' property ''discreteOutput'' must be passed to the function' ] )
	
end% done with givenInputs(5)

if givenInputs(6)
	
	%validating globalThresholdValue
	
	%validating that it's a single value
	valid = true;
	if ~isscalar( globalThresholdValue )
		valid = false;
	end
	
	if ~valid
		error( [ 'the global threshold should be a scalar value' ...
			', i.e. a 1x1 vector' ] );
	end
	
	%validating the value is inside the proper range
	if globalThresholdValue >= 0
		%good to go
	else
		error( [ 'global threshold for the MLMVN algorithm ' ...
			'must be greater than or equal to 0' ] )
	end
	
	clear valid s
else
    if ~strcmp(stoppingCriteria, 'test')
        error( ['property ''globalThresholdValue'' not passed' ...
		' to the function' ] );
    end
end

if givenInputs (7)%localThresholdValue
	
	%validating localThresholdValue

	%making sure it's a single value
	valid = true;
	if ~isscalar( localThresholdValue )
		valid = false;
	end
	
	if ~valid
		error( [ 'the local threshold should be a scalar value' ...
			', i.e. a 1x1 vector' ] );
	end
	
	
	%making sure it fits the appropriate value range
	if localThresholdValue >= 0 
		
		if (localThresholdValue <= globalThresholdValue) 
			% fits the range (0,globalThresholdValue]
        else
            if ~strcmp(stoppingCriteria, 'test')
                error( ['the local threshold value input should ' ...
				'be less than the global threshold value' ] );
            end
		end
		
	else
		error( ['threshold values for calculation of the end of ' ...
			'learning must be non-negative' ] );
	end
	
	clear valid s
	
else% ltv not passed to the function
	error( '''localThresholdValue'' not passed to the function' );
end

if givenInputs(8)%network
	
	%a network should be passed as a cell array
	if iscell(network) && isvector(network)
		%great
	else
		error(['a network should be a cell array of layers'...
			' of neurons, even a network with one layer'] )
	end
	
else
	% the following is probably a redundant check
	if ~givenInputs(1)
		error( ['the MLMVN algorithm must be pased either'...
			' an existing network or the desired dimensions '...
			'for a newly created network' ] );
	end
end

if givenInputs(9)%initialWeights
	
	%validating initialWeights
	
	%making sure that its a string
	if ~ischar( initialWeights )
		error( [ 'input initialWeights should be a string indicating'...
			' the desired method for initialization of the weights' ]);
	end
	
	if sum( strcmp( acceptableInitialWeightsValues, initialWeights ) ) > 0
		%desired situation
	else% bad input
		
		%informing the user of bad input
		s = sprintf( 1, [ '\n\nInput ''initialWeights must be one of a set of'...
			'strings. The following are acceptable strings: ' ] );
		len = length( acceptableInitialWeightsValues );
		s = [s sprintf( 1, '\n%s\n', acceptableInitialWeightsValues{1:len} ) ];
		
		%ending the function
		error( 'MLMVN:badInput',s );
	end
else
	if ~givenInputs(8)
		s = '\n\n ''initialWeights'' was not given as input to the function';
		s = [ s ' and is not an optional input\n\n' ];
		error( 'MLMVN:BadInput', s ); 
	end
end

if givenInputs(10)
	%numberOfSectors
	if isnumeric( numberOfSectors ) && isscalar( numberOfSectors ) && ~iscell( numberOfSectors )
		if ( floor(numberOfSectors) ~= numberOfSectors )
			s = 'number of sectors must be a whole number';
			error( 'MLMVN:BadInput', s );
		end	
	else
		error( 'MLMVN:BadInput', '\nbad input: numberOfSectors\n\n' );
	end
else
	if discreteInput || discreteOutput
		s = 'numberOfSectors needed as input in the case of';
		s = [s  ' discrete input or output' ];
		error( 'MLMVN:BadInput', s );
	end
end

%done with numberOfSectors

if givenInputs(11)
	
	a = angularGlobalThresholdValue;
	if ~iscell(a) && isnumeric(a) && isscalar(a)
		if a < 0
			s = 'angularGlobalThresholdValue must be ';
			s = [s 'greater than 0'];	
			error('MLMVN:BadInput',s);
		end
	else
		s = 'angularGlobalThresholdValue must be a scalar numeric value';
		s = [ '\n\n' s '\n\n' ];
		error('MLMVN:BadInput',s);
	end
else
	if SoftMargins
		s = '\n\angularGlobalThresholdValue needed ';
		s = [s 'for angular rmse learning\n\n' ];
		error( 'MLMVN:BadInput', s );
	end
end

if givenInputs(12)
	
	a = angularLocalThresholdValue;
	if ~iscell(a) && isnumeric(a) && isscalar(a)
		if a < 0
			s = 'angularLocalThresholdValue must be ';
			s = [s 'greater than 0'];	
			error('MLMVN:BadInput',s);
		end
	else
		s = 'angularLocalThresholdValue must be a scalar numeric value';
		s = [ '\n\n' s '\n\n' ];
		error('MLMVN:BadInput',s);
	end
else
	if SoftMargins
		s = '\n\angularLocalThresholdValue needed ';
		s = [s 'for angular rmse learning\n\n' ];
		error( 'MLMVN:BadInput', s );
	end
end

if givenInputs(14)
	x = findall(0);
	if x(x==figureHandle) == 0
		s = '\n\ninvalid figure handle passed to MLMVN\n\n';
		error( 'MLMVN:BadInput', s );
	else
		iterHandle = findobj('tag','iterationText');
		learnTextHandle = findobj('tag','learningText');
		learnButtonHandle = findobj('tag','pushButton2');
		
		set(learnTextHandle,'string','Learning - Started');
	end
	calledByFig = true;
else
	calledByFig = false;
end

% *** END OF INPUT VALIDATION, BEGINNING NECESSARY
% *** VARIABLE INITIALIZATION

%storing the value 2pi instead of having to multiply to get it every time
pi2 = 2*pi;

% a variable, which is equal to a half of Number of sectors; floor is
% needed in NumberOfSectors is odd
numberOfSectorsHalf=floor(numberOfSectors/2);

%generation of the bisectors' angular values
if SoftMargins
	temp = 0:numberOfSectors-1;
	bisectors = pi2*(temp+.5)/numberOfSectors;
	clear temp;
end


% sectorsize is the angular size of one sector for a discrete output
if discreteOutput
	sectorSize = pi2/numberOfSectors;
end

% Generation of complex numbers - roots of unity on the sectors' borders
% They'll be contained in the array Sectors
Sector=zeros(1,numberOfSectors);
for jj=1:numberOfSectors
    angSector=pi2*(jj-1)/numberOfSectors;
    Sector(jj)=exp(1i*angSector);
end
clear angSector;

% if network was given, then its size must be read
if givenInputs(8)
    [sizeOfMmlvn1, sizeOfMlmvn2] = cellfun(@size,network);
    sizeOfMlmvn = sizeOfMmlvn1;
end

%initializing variable 'numberOfOutputs'
numberOfOutputs = sizeOfMlmvn(end);


%initializing the variable 'inputsPerSample' which is the number of input
%values given for each learning sample in the matrix 'inputs' and this
%value is equal to 'columns(inputs) - numberOfOutputs' since each row 
%first consists of the inputs for the sample followed by the outputs 
%of the sample
[rowsInputs colsInputs] = size( inputs );
% if there is a discrete output and multiple output neurons, we consider
% that the network has a single output for the testing puposes
if (discreteOutput == 1) && (numberOfOutputs > 1) && strcmp(stoppingCriteria, 'test')
    inputsPerSample = colsInputs - 1;
else
    inputsPerSample = colsInputs - numberOfOutputs;
end

%grabbing the columns containing the desired outputs of the MLMVN
desiredOutputs = inputs(1:rowsInputs,inputsPerSample+1:end );

%ridding matrix 'inputs' of values now stored in 'desiredOutputs'
inputs = inputs(:,1:inputsPerSample);

%updating these values because of the change in size of outputs
[rowsInputs colsInputs] = size( inputs );

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
% preallocatiFng a matrix for correction of those errors, which  jump over
% pi (or half of the number of sectors)
mask = networkOutputs;
[n1,n2] = size(mask);

%initializing the variable that will hold the global error of the 
% network for each input sample
networkErrors = networkOutputs(1:end,1);
if SoftMargins
	netAngErrors = networkErrors;
end

%initializing the neural network's weights if necessary
if ~givenInputs(8)
	
	len = length( sizeOfMlmvn );
	
	%preallocating the number of cells. network ends up being a 1xN
	% cell vector where N is the number of layers
	network = cell( 1, len );
	%beginning creation of the layers
	
	%creating layer 0
	
	% ( I would just copy the following in the if part of the if else for
	% each case of possible initialWeights values modifying only the 
	% lines where values are created to fit the desired method
	if sum( strcmp( initialWeights, 'random' ) ) > 0
		
        % Re and Im parts of random weights are in the range [-0.5,0.5]
        
		%each row of the network is the weights of a single neuron and neurons
		%in the first layer need the number of inputs per sample + 1 weights
		network{1} = rand( [ sizeOfMlmvn(1), inputsPerSample+1 ] )-0.5 + ...
			(rand( [ sizeOfMlmvn(1),inputsPerSample+1 ] )-0.5) .* 1i;

		%creating subsequent layers of the network if necessary
		if len > 1
			for ii = 2:len
				%for the following layers, the number of weights used is as
				%seen below: number of neurons of the previous layer + 1
				network{ii} = rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5 ...
					+ (rand( [ sizeOfMlmvn(ii), sizeOfMlmvn(ii-1)+1 ] )-0.5) .* 1i;
			end
        
			
			clear ii
		end
	
	end%end if
	
	clear len
	
end% end of initialization of network

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

% a desired discrete output equals a root of unity corresponding to the
% bisector of a desitred sector
if discreteOutput
	ComplexValuedDesiredOutputs = exp( (desiredOutputs+.5)*1i*sectorSize );
	learnAo = 0;
    % AngularDesiredOutputs - arguments of the discrete desired outputs
    AngularDesiredOutputs = mod(angle(ComplexValuedDesiredOutputs), pi2);
else
    % AngularDesiredOutputs - arguments of the continuous desired outputs
    AngularDesiredOutputs = desiredOutputs;
    ComplexValuedDesiredOutputs = exp( (desiredOutputs)*1i);
end

if discreteInput
	% converting sector values (which are integers) into 
    % corresponding complex numbers located on the unit circle 
	% argumetnts of inputs
    theta = pi2 .* (inputs) ./ numberOfSectors;
	
    % Re and Im parts of inputs
	[re, im] = pol2cart( theta, ones( rowsInputs, colsInputs ) );
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





%% END OF VARIABLE INITIALIZATION

%% DATA PROCESSING
%-----------------------------------------------------------
%ENTERING DATA PROCESSING STAGE 
%-----------------------------------------------------------

desiredAlgo = 0;
%checking for desired algorithm based on stopping criteria

if strcmp( stoppingCriteria, 'rmse' )
		desiredAlgo = 1;
elseif strcmp( stoppingCriteria, 'test' )
        desiredAlgo = 2;
else
	%not yet implemented stoppingCriteria algorithm
end

%if somehow desiredAlgo stayed at 0
if desiredAlgo < 1
	s = [ '\n\nerror: a not yet implemented algorithm'...
		' was requested\n\n' ];
	error( 'MLMVN:NotImplementedAlgorithm', s )
end
% or if it accidentally has some weird value
if desiredAlgo > 5
	s = '\n\nerror: something unexpected has occurred\n\n';
	error( 'MLMVN:ExecutionError', s );
end

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
                				
				neuronOutputs{ii} = ...
					network{ii}(:,2:end) * (inputs(aa,:)).' + ...
					network{ii}(:,1);
              

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

					neuronOutputs{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);
					
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

					neuronOutputs{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);
					
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
				networkOutputs(aa,:) = ...
					neuronOutputs{ii};
				
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
                				
				weightedSum{ii} = ...
					network{ii}(:,2:end) * (inputs(aa,:)).' + ...
					network{ii}(:,1);
              

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

					weightedSum{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);
					
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

					weightedSum{ii} = ...
						network{ii}(:,2:end) * neuronOutputs{ii-1} + ...
						network{ii}(:,1);
					
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
				networkOutputs(aa,:) = ...
					neuronOutputs{end};

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
                            neuronErrors{ii} (jjj) = ComplexValuedDesiredOutputs(aa,jjj).' - ...
								neuronOutputs{ii}(jjj);

                % finally we obtain the output neurons' errors
				% normalizing the global errors (dividing them
				% by the (number of neurons in the preceding
				% layer+1)
				neuronErrors{ii} = ...
					neuronErrors{ii} ./ (sizeOfMlmvn(ii-1)+1);
						
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
								neuronErrors{ii} = ( ...
									temp * neuronErrors{ii+1}...
									 ) ./ (sizeOfMlmvn(ii-1)+1);
                            else % to the 1st hidden layer
								neuronErrors{ii} = ( ...
									temp * neuronErrors{ii+1} ...
									 ) ./ (inputsPerSample+1);
                            end % end of the if statement
                        end % end of the for lop ii over the layers
						%**************************************************
						% *** NOW CORRECTING THE WEIGHTS OF THE NETWORK ***
						
						%handling the 1st hidden layer
                            
                            % learning rate is a reciprocal absolute value
                            % of the weighted sum
							learningRate = ( 1 ./ abs( weightedSum{1} ) );
                          % all weights except bias (w0 = w(1) in Matlab)
                  
							network{1}(:,2:end) = network{1}(:,2:end) + (learningRate .* ...
								neuronErrors{1}) * conj(inputs(aa,:));
						    % bias (w0 = w(1) in Matlab)
							network{1}(:,1) = network{1}(:,1) + learningRate ...
								.* neuronErrors{1};

						%correcting following layers
						for ii = 2:numberOfLayers
							
							%**********************************************
							%calculating new output of preceding layer
                            if (ii==2) % if a preceding layer is the 1st one
                                weightedSum{1} = ...
                                    network{1}(:,2:end) * (inputs(aa,:)).' + ...
                                    network{1}(:,1);
                            else % if a preceding layer is not the 1st one
                                weightedSum{ii-1} = ...
                                    network{ii-1}(:,2:end) * neuronOutputs{ii-2} + ...
                                    network{ii-1}(:,1);
                            end
					
					%APPLYING CONTINUOUS ACTIVATION FUNCTION
					%TO THE FIRST HIDDEN LAYER NEURONS
					%% CONTINUOUS OUTPUT CALCULATION
					%output=weighted sum/abs(weighted sum)
					neuronOutputs{ii-1} = weightedSum{ii-1} ./ abs(weightedSum{ii-1});							
						
							%**********************************************
                            % learning rate is a reciprocal absolute value
                            % of the weighted sum					
							learningRate = ( 1 ./ abs( weightedSum{ii} ) );
							
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

						        % bias (w0 = w(1) in Matlab)	
								network{ii}(:,1) = network{ii}(:,1) + learningRate .* ...
									neuronErrors{ii};
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

                                % bias (w0 = w(1) in Matlab)
								network{ii}(:,1) = network{ii}(:,1) + ...
									neuronErrors{ii};
                            end % end of if ii < numberOfLayers statement
                        end %end of ii for loop over the layers
                    end % end of "if check" 
					%done with correction of weights for current sample

					%**************************************************
		
				end% end of aa for loop
				
			end% end of if statement for ~finishedLearning
		
			%% END OF LEARNING
		
		end% end of '~finishedLearning' while loop	
		
		%% END OF ANGULAR RMSE ALGORITHM	
        
        
	
       elseif desiredAlgo == 2 %TESTING - In this branch only testing is implemented
         
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
				
				neuronOutputs{ii} = ...
					network{ii}(:,2:end) * ( inputs(aa,:) ).' + ...
					network{ii}(:,1);

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
                          mask(i3) = numberOfSectors;
                          errors(i3) = mask(i3)-errors(i3);
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
            
            
		
	else% something weird happened, control is not supposed to reach here
		s = 'desiredAlgo has somehow acquired a strange value';
		s = [ '\n\n' s '\n\n' ];
		error('MLMVN:ControlPathError', s);
	end


%% RETURNING OF RESULTS

if calledByFig
	
	x = getappdata(figureHandle,'userdata');
	x.currentlyLearning = false;
	setappdata(figureHandle,'userdata',x);
	
	if finishedLearning
		set(iterHandle,'string',num2str(iterations));
		set( learnTextHandle, 'string', 'Learning - Converged' );
	end
	
end

%copying the weights of the network to the output variable
output.network = network;
if (strcmp(stoppingCriteria,'test')) % output the evaluation results for testing
    output.DesiredOutputs = desiredOutputs;
    output.NetworkOutputs = networkOutputs;
    output.AbsoluteError = AbsErrors;
    output.Accuracy = Accuracy;
    output.RMSE = rmse;
else
    output.iterations = iterations; % output: the number of iterations for learning
end
end% end of function