% To be run after training a neural network (neural_network_with_momentum), and inputs are selected
% Calculates the average time required to classify an input

testTimes = [];

allInputs = [trainingInputs testInputs];
allTargets = [trainingTargets testTargets];

% Loop through 
for i = 1 : length(allInputs)
    tic

    % initalize
    for layer=1:LAYERS
        Z = zeros(NODES(layer),1);

        if layer == 1           % input layer 
            output{layer} = [BIAS; allInputs(:, i)]; 
        elseif layer == LAYERS  % output layer
            output{layer} = Z;     
        else                    % hidden layers
            output{layer} = [BIAS; Z]; 
        end

        tot{layer} = Z;
        delta{layer} = Z;
    end

    % forward path
    for layer = 2:LAYERS                                % layer working on
        tot{layer} = weights{layer}*output{layer-1};    % performs sum
        result = activation_function(tot{layer});       % get output of neuron

        if layer ~= LAYERS
            output{layer} = [output{layer}(1); result]; % add bias output back in
        else
            output{layer} = result;
        end            
    end
    
    testTimes = [testTimes toc];
end

avgTimeMs = mean(testTimes) * 1000;
disp(['Average Time to Classify: ', num2str(avgTimeMs), ' ms']);