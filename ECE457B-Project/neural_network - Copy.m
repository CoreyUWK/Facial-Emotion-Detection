% questions 
% 1: set up nnstart does not allow for multiple hidden layers
% 2: output values are in range 0 to 1 should they be passed to an
% activation function for a binary result
% 3: need to complete code for neural network 
% 4: need to complete libary code for matlab neural network
% 5: need to compare results

clear;
format longg;

activation_function = @(tot) (1 ./ (1 + exp(-1.*tot)));

HIDDEN_LAYERS = 1;
LAYERS = HIDDEN_LAYERS + 2;
NODES = [136, 10, 4]; % total number of nodes (without bias in layer)
%NODES = [2, 2, 1];

E_max = 0.01;
eta = 0.2;
E = Inf;
k = 1;
BIAS = -1;
epoch = 0;

% initalize weights
for layer=2:LAYERS 
    %weights{layer} = ones(NODES(layer),NODES(layer-1) + 1).*0.2;
    weights{layer} = rand(NODES(layer),NODES(layer-1) + 1)-0.5; % +1 for BIAS in previous layer
end

% load data for input and targets
load('N:\ECE 457B\ECE457B-Project\Dataset.mat');
inputs = allLandmarks;
targets = outputEmotions;

% x1 = [0.3; 0.4];
% x2 = [0.1; 0.6];
% x3 = [0.9; 0.4];
% inputs = [x1, x2, x3];
% targets = [0.88 0.2; 0.82 0.3; 0.57 0.5];

% apply input patterns (one epoch)
[input_number, exemplars] = size(inputs);
while (E >= E_max)  % loop to check error
    E = 0;
    epoch = epoch + 1;
    
    for k = 1:exemplars % loop over input examples
        
        % initalizate
        for layer=1:LAYERS
            Z = zeros(NODES(layer),1);
            
            if layer == 1 % input layer 
                output{layer} = [BIAS; inputs(:, k)]; 
            elseif layer == LAYERS % output layer
                output{layer} = Z;     
            else % hidden layers
                output{layer} = [BIAS; Z]; 
            end
            
            tot{layer} = Z;
            delta{layer} = Z;   
        end

        % forward path
        for layer = 2:LAYERS % layer working on
            tot{layer} = weights{layer}*output{layer-1}; % performs sum
            result = activation_function(tot{layer});
            
            if layer ~= LAYERS
                output{layer} = [output{layer}(1); result]; % add bias output back in
            else
                output{layer} = result;
            end            
        end

        % compute and update cumulative error per epoch
        E_k = sum(0.5 .* (targets(:, k) - output{LAYERS}).^2); % check
        E = E + E_k;

        % caluclate change in weights 
        % backward path
        for layer = LAYERS:-1:2 % layer working on
            if layer==LAYERS
                delta{layer} = -1.*(targets(k)-output{layer}) .* ((1-output{layer}).*output{layer});
                %delta_weight{layer} = -1.*eta.*delta{layer}.*output{layer-1};
                %weights{layer} = (weights{layer}' + delta_weight{layer})';
            else                              
                delta{layer} = ((1-output{layer}(2:end)).*output{layer}(2:end)) .* (weights{layer+1}(:,2:end)'*delta{layer+1}); % check if need to sum second part
            end
            delta_weight{layer} = -1.*eta.*(delta{layer}*output{layer-1}');
            weights{layer} = weights{layer} + delta_weight{layer};
        end
    end
    
    clc;
    disp('--------------------------');
    disp(['Epoch = ', num2str(epoch)]);
    disp(['Error = ', num2str(E_k)]);
    disp(['Error = ', num2str(E)]);
    disp('----------------------');
end