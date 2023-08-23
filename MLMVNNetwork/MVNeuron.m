classdef MVNeuron
    properties
        Weights = []
    end
    methods % Begin Methods
        % Begin Constructor
        function obj = MVNeuron(numInputs) 
            xmin = -0.5;
            xmax = 0.5;
            n = numInputs + 1;
            obj.Weights = (xmin+rand(1,n)*(xmax-xmin)) + (xmin+rand(1,n)*(xmax-xmin))*1i;
        end 
        % End Constructor
        
        % Activation Function
        function output = activate(obj, data) % MVN Continuous Activation
           output = MVContinuousActivation(obj.Weights, data);
        end
        % End Activation Function
        
    end % End Methods
end