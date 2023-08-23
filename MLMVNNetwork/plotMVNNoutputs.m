function [] = plotMVNNoutputs(outputs, expectedOutputs)
    x1 = linspace(0, 1, length(outputs));
    x2 = linspace(0, 1, length(expectedOutputs));
    
    scatter(x1, outputs)
    hold on;
    scatter(x2, expectedOutputs)
    hold off
    xlabel('Sample')
    ylabel('Outputs')
end 
