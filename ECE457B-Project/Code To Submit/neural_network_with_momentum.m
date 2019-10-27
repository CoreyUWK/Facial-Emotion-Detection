clear;
clc;
format longg;

% Sigmoid activation function
activation_function = @(tot) (1 ./ (1 + exp(-1.*tot)));

% Select Which Landmarks to Use
useJaw = true;
useEyebrows = true;
useEyes = true;
useNose = true;
useMouth = false;
useAll = false;

sequence_x = [];
sequence_y = [];

% Jaw
if useJaw
    sequence_x = [sequence_x, 1:17];
    sequence_y = [sequence_y, 1+68:17+68];
end
% Eyebrows
if useEyebrows
    sequence_x = [sequence_x, 18:27];
    sequence_y = [sequence_y, 18+68:27+68];
end
% Nose
if useNose
    sequence_x = [sequence_x, 28:36];
    sequence_y = [sequence_y, 28+68:36+68];
end
% Eyes
if useEyes
    sequence_x = [sequence_x, 37:48];
    sequence_y = [sequence_y, 37+68:48+68];
end
% Mouth
if useMouth
    sequence_x = [sequence_x, 49:68];
    sequence_y = [sequence_y, 49+68:68+68];
end

% All Landmarks
if useAll
    sequence_x = 1:68;
    sequence_y = 69:136;
end

sequence = [sequence_x, sequence_y];

% Neural Network Parameters
NODES = [length(sequence), 50, 4]; % total number of nodes (without bias in layer)
LAYERS = length(NODES);
HIDDEN_LAYERS = LAYERS - 2;

E_max = 0.01;        % max % error to minimize to 
maxEpochs = 5000;   % max epochs to minimize
eta = 1e-3;         % Learning Rate
gamma = 1e-5;       % Momentum Term, set to 0 for no momentum
E = Inf;            % initalize to large value

k = 1;
BIAS = 1;
epoch = 0;
E_percent_train = 100;
E_percent_test = 100;

plotInterval = 10;  % interval of number of epoch till data is ploted

% initalize weights
for layer=2:LAYERS 
    weights{layer} = (rand(NODES(layer),NODES(layer-1) + 1)-0.5)*(1e-1); % +1 for BIAS in previous layer
end

% load data for input and targets
load('Dataset.mat');
[totalInputs, allExemplars] = size(allLandmarks(sequence,:));

numTraining = round(allExemplars*0.65); % use 65% of exemplars for training
numTest = allExemplars - numTraining;

% calculate means across single axis and subtract them for test data
trainingMean = mean(allLandmarks(sequence,1:numTraining),2)*ones(1,length(allLandmarks(sequence,1:numTraining)));
trainingInputs = allLandmarks(sequence,1:numTraining) - trainingMean;

% calculate means across single axis and subtract them for test data
testMean = mean(allLandmarks(sequence,numTraining+1:end),2)*ones(1,numTest);
testInputs = allLandmarks(sequence,numTraining+1:end) - testMean;

[~, trainingExemplars] = size(trainingInputs);
[~, testExemplars] = size(testInputs);

% get x and y input indices
xIndices = 1:length(sequence_x);
yIndices = (1:length(sequence_y)) + length(xIndices);

% Subtract x and y means for training 
for exampleIdx = 1 : trainingExemplars
    trainingInputs(xIndices,exampleIdx) = trainingInputs(xIndices,exampleIdx) - mean(trainingInputs(xIndices,exampleIdx));
    trainingInputs(yIndices,exampleIdx) = trainingInputs(yIndices,exampleIdx) - mean(trainingInputs(yIndices,exampleIdx));
end

% Subtract x and y means for testing 
for exampleIdx = 1 : testExemplars
    testInputs(xIndices,exampleIdx) = testInputs(xIndices,exampleIdx) - mean(testInputs(xIndices,exampleIdx));
    testInputs(yIndices,exampleIdx) = testInputs(yIndices,exampleIdx) - mean(testInputs(yIndices,exampleIdx));
end

trainingTargets = outputEmotions(:,1:numTraining);
testTargets = outputEmotions(:,numTraining+1:end);
trainingTargetsInd = vec2ind(trainingTargets);
testTargetsInd = vec2ind(testTargets);

plotEpochs = [];
E_stored = [];
Matches_stored = [];

% Initialize delta weights for momentum
for layer=2:LAYERS
    delta_weight{layer} = zeros(NODES(layer), NODES(layer-1) + 1);
end

while (E >= E_max && epoch < maxEpochs)
% while (true)  
    E = 0;     % reset error
    trainingOutput = zeros(4,numTraining);
    epoch = epoch + 1;
    
    % initalize weights per epoch for network
    for layer=2:LAYERS
        prev_delta_weight{layer} = delta_weight{layer};
        delta_weight{layer} = zeros(NODES(layer), NODES(layer-1) + 1);
    end
    
    for k = 1:trainingExemplars % loop over input examples
        % initalizate
        for layer=1:LAYERS
            Z = zeros(NODES(layer),1);
            
            if layer == 1           % input layer 
                output{layer} = [BIAS; trainingInputs(:, k)]; 
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
            
            if layer ~= LAYERS                              % if not last layer
                output{layer} = [output{layer}(1); result]; % add bias output back in
            else
                output{layer} = result;
            end            
        end
        
        % store output
        [maxVal, maxInd] = max(output{LAYERS});
        trainingOutput(maxInd,k) = 1;

        % compute and update cumulative error over an epoch (offline)
        E_k = sum(0.5 .* (trainingTargets(:, k) - output{LAYERS}).^2); % check
        E = E + E_k;

        % caluclate change in weights 
        % backward path
        for layer = LAYERS:-1:2 % layer working on
            if layer==LAYERS
                delta{layer} = -1.*(trainingTargets(:,k)-output{layer}) .* ((1-output{layer}).*output{layer});
            else                              
                delta{layer} = ((1-output{layer}(2:end)).*output{layer}(2:end)) .* (weights{layer+1}(:,2:end)'*delta{layer+1}); % check if need to sum second part
            end
            delta_weight{layer} = delta_weight{layer} + (delta{layer}*output{layer-1}');            
            % weights{layer} = weights{layer} + delta_weight{layer};
        end
    end
    
    if epoch == 1
        E_Top_train = E;  % for percentage
    end
    E_percent_train = (E/E_Top_train)*100;
      
    % batch learning = ofline learning
    for layer=2:LAYERS
        weights{layer} = weights{layer} + (-1.*eta.*delta_weight{layer}) + gamma.*prev_delta_weight{layer};
    end
    
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
        
        [maxVal, maxInd] = max(output{LAYERS});
        testOutput(maxInd,k) = 1;

        % compute and update cumulative error per epoch
        E_k = sum(0.5 .* (testTargets(:, k) - output{LAYERS}).^2); % check
        E_test = E_test + E_k;
    end
    
    if epoch == 1
        E_Top_test = E_test;  % for percentage
    end
    E_percent_test = (E_test/E_Top_test)*100;
    
    disp(['Epoch: ', num2str(epoch) ' E: ', num2str(E) ' E_test: ', num2str(E_test)]);
    
    E_stored = [E_stored; E E_test;];
    %E_stored = [E_stored; E_percent_train E_percent_test;];
    trainingMatches = length(find(vec2ind(trainingOutput) == trainingTargetsInd));
    testMatches = length(find(vec2ind(testOutput) == testTargetsInd));
    Matches_stored = [Matches_stored; trainingMatches/trainingExemplars  testMatches/testExemplars;];
    
    if(mod(epoch-1,plotInterval) == 0)
        plotEpochs = 1:epoch;

        figure(1);
        plot(plotEpochs, E_stored);
        
        figure(2);
        plot(plotEpochs, Matches_stored);
    end
end

% Plot Results
epoch
lastEpoch = min(epoch,5000);
plotEpochs = 1:lastEpoch;

% Plot 1: Training Error vs Test Error
figure(1);
plot(plotEpochs, E_stored(plotEpochs,:), 'LineWidth', 2);
set(gcf,'Position',[100 500 640 480]);
legend('Training', 'Test', 'Location', 'NorthEast');
title('Cumulative Error', 'FontSize', 15);
xlabel('Epochs');
xlim([0 lastEpoch]);
ylabel('Error');

% Plot 2: Training Matches percent and Test Matches percent
figure(2);
plot(plotEpochs, Matches_stored(plotEpochs,:)*100, 'LineWidth', 2);
set(gcf,'Position',[800 500 640 480]);
title('Classification Results', 'FontSize', 15);
legend('Training', 'Test', 'Location', 'SouthEast');
xlabel('Epochs');
xlim([0 lastEpoch]);
ylabel('Correctly Classified (%)');
ylim([0 102]);

Matches_stored(lastEpoch,:)

% Plot 3: Confusion Plots for Training and Testing
figure(3);
title('Training Confusion Matrix');
plotconfusion(trainingTargets, trainingOutput, 'Training')

figure(4);
title('Test Confusion Matrix');
plotconfusion(testTargets, testOutput, 'Test')