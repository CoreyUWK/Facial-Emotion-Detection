clear;
format longg;

activation_function = @(tot) (1 ./ (1 + exp(-1.*tot)));

NODES = [136, 300, 4]; % total number of nodes (without bias in layer)
LAYERS = length(NODES);
HIDDEN_LAYERS = LAYERS - 2;

E_max = 5;         % max % error to minimize to 
eta = 1e-3;        % Learning Rate
E = Inf;           % initalize to large value
k = 1;
BIAS = 1;    
epoch = 0;
E_percent = 100;   

% initalize weights
for layer=2:LAYERS 
    weights{layer} = (rand(NODES(layer),NODES(layer-1) + 1)-0.5)*(1e-1);  % +1 for BIAS in previous layer
end

% load data for input and targets
load('Dataset.mat');

% readjust the input landmark value range to center it around zero (center)
% of multi-axis 
inputs = allLandmarks(:,1:200) - (mean(allLandmarks(:,1:200),2)*ones(1,length(allLandmarks(:,1:200))));  
[input_number, exemplars] = size(inputs);
    
% Subtract means
xIndices = 1:input_number/2;
yIndices = input_number/2+1 : input_number;

% This might be incorrect as subracting the mean on different axises (x1
% and x2 are not of the same input type so subtracting mean of them is
% incorrect
for exampleIdx = 1 : exemplars
    inputs(xIndices,exampleIdx) = inputs(xIndices,exampleIdx) - mean(inputs(xIndices,exampleIdx));
    inputs(yIndices,exampleIdx) = inputs(yIndices,exampleIdx) - mean(inputs(yIndices,exampleIdx));
end

targets = outputEmotions(:,1:200);

% apply input patterns (one epoch)
while (E_percent >= E_max)  % loop to check error
    E = 0;          % reset error
    epoch = epoch + 1;  % update epoch 
    
    % initalize weights per epoch for network
    for layer=2:LAYERS
        delta_weight{layer} = zeros(NODES(layer), NODES(layer-1) + 1);
    end
    
    for k = 1:exemplars % loop over input examples
        % initalizate
        for layer=1:LAYERS
            Z = zeros(NODES(layer),1);
            
            if layer == 1           % input layer 
                output{layer} = [BIAS; inputs(:, k)]; 
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

        % compute and update cumulative error over an epoch (offline)
        E_k = sum(0.5 .* (targets(:, k) - output{LAYERS}).^2); 
        E = E + E_k;

        % caluclate change in weights 
        % backward path
        for layer = LAYERS:-1:2 % layer working on
            if layer==LAYERS
                delta{layer} = -1.*(targets(:,k)-output{layer}) .* ((1-output{layer}).*output{layer});
            else                              
                delta{layer} = ((1-output{layer}(2:end)).*output{layer}(2:end)) .* (weights{layer+1}(:,2:end)'*delta{layer+1}); % check if need to sum second part
            end
            delta_weight{layer} = delta_weight{layer} + (delta{layer}*output{layer-1}');           
            % weights{layer} = weights{layer} + delta_weight{layer};
        end
    end
    
    if epoch == 1
        E_Top = E;  % for percentage
    end
    E_percent = (E/E_Top)*100;
    
    % batch learning = ofline learning
    for layer=2:LAYERS
        weights{layer} = weights{layer} + (-1.*eta.*delta_weight{layer});
    end
    
    clc;
    disp('--------------------------');
    disp(['Epoch = ', num2str(epoch)]);
    disp(['Error = ', num2str(E_k)]);
    disp(['Error = ', num2str(E)]);
    disp(['% Error = ', num2str(E_percent)]);
    disp('----------------------');
end