testTimes = [];

allInputs = [trainingInputs testInputs];
allTargets = [trainingTargets testTargets];

for i = 1 : 225
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

totalTimes = [];
for i = 25:25:225
    totalTimes = [totalTimes; sum(testTimes(1:i)) * 1000];
end