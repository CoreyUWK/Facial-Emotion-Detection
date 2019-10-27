
load('weigths_size.mat');
activation_function = @(tot) (1 ./ (1 + exp(-1.*tot)));

NODES = [136, 50, 4]; % total number of nodes (without bias in layer)
LAYERS = length(NODES);
HIDDEN_LAYERS = LAYERS - 2;

E_max = 0.01;          % max % error to minimize to
maxEpochs = 5000;  % max epochs to minimize
eta = 1e-3;         % Learning Rate
E_test = Inf;
k = 1;  
BIAS = 1;
epoch = 0;

% load data for input and targets
load('Dataset.mat');
[totalInputs, allExemplars] = size(allLandmarks);

numTraining = round(allExemplars*0.65); % use 65% of exemplars for training
numTest = allExemplars - numTraining;

% calculate means across single axis and subtract them for test data
testMean = mean(allLandmarks(:,numTraining+1:end),2)*ones(1,numTest); % mean from test data
testInputs = allLandmarks(:,numTraining+1:end) - testMean;

[~, testExemplars] = size(testInputs);

% get x and y input indices
xIndices = 1:totalInputs/2;
yIndices = totalInputs/2+1 : totalInputs;

% Subtract x and y means for testing 
for exampleIdx = 1 : testExemplars
    testInputs(xIndices,exampleIdx) = testInputs(xIndices,exampleIdx) - mean(testInputs(xIndices,exampleIdx));
    testInputs(yIndices,exampleIdx) = testInputs(yIndices,exampleIdx) - mean(testInputs(yIndices,exampleIdx));
end

plotEpochs = [];
E_stored = [];
Matches_stored = [];

E_test = 0;
testOutput = zeros(4,numTest);
for k = 1:testExemplars  % loop over test cases
    % initalize
    for layer=1:LAYERS
        Z = zeros(NODES(layer),1);

        if layer == 1           % input layer 
            output{layer} = [BIAS; testInputs(:, k)]; 
        elseif layer == LAYERS  % output layer
            output{layer} = Z;     
        else                    % hidden layers
            output{layer} = [BIAS; Z]; 
        end

        tot{layer} = Z;
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

    [maxVal, maxInd] = max(output{LAYERS});
    testOutput(maxInd,k) = 1;
end