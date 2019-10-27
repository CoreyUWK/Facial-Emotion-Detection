% clear;
% clc;
% close all;
% format compact
% 
% load('Dataset.mat');
% x = allLandmarks;
% t = outputEmotions;
x = [trainingInputs testInputs];
t = [trainingTargets testTargets];

[numRows numCols] = size(x);

% nNeurons = [5, 20, 50];
nNeurons = [20];
nLayers = 1;

accuracyVals = zeros(length(nNeurons),2);

% Test different number of neurons per layer
for n = 1:length(nNeurons)
    % Create a Pattern Recognition Network
    net = patternnet(repmat(nNeurons(n),1,nLayers));

    % Choose Input and Output Pre/Post-Processing Functions
    % For a list of all processing functions type: help nnprocess
    net.input.processFcns = {'removeconstantrows','mapminmax'};
    net.output.processFcns = {'removeconstantrows','mapminmax'};


    % Setup Division of Data for Training, Validation, Testing
    % For a list of all data division functions type: help nndivide
    net.divideFcn = 'dividerand';  % Divide data randomly
    net.divideMode = 'sample';  % Divide up every sample
    net.divideParam.trainRatio = 0.60;
    net.divideParam.valRatio = 0.05;
    net.divideParam.testRatio = 0.35;

    % For help on training function 'trainscg' type: help trainscg
    % For a list of all training functions type: help nntrain
    net.trainFcn = 'trainscg';  % Scaled conjugate gradient

    % Choose a Performance Function
    % For a list of all performance functions type: help nnperformance
    net.performFcn = 'crossentropy';  % Cross-entropy

    % Choose Plot Functions
    % For a list of all plot functions type: help nnplot
    net.plotFcns = {'plotperform','plottrainstate','ploterrhist', ...
      'plotregression', 'plotfit'};

    % Train the Network
    [net,tr] = train(net,x,t);

    % Test the Network
    y = net(x);
    e = gsubtract(t,y);
    tind = vec2ind(t);
    yind = vec2ind(y);
    percentErrors = sum(tind ~= yind)/numel(tind);
    performance = perform(net,t,y);

    % Recalculate Training, Validation and Test Performance
    trainTargets = t .* tr.trainMask{1};
%     valTargets = t  .* tr.valMask{1};
    testTargets = t  .* tr.testMask{1};
%     trainPerformance = perform(net,trainTargets,y);
%     valPerformance = perform(net,valTargets,y);
%     testPerformance = perform(net,testTargets,y);

    % View the Network
    view(net);

    % Plots
    % figure, plotconfusion(t,y);
    % figure, plotconfusion(testTargets,y);
%     plotconfusion(trainTargets,y);
    [c,cm,ind,per] = confusion(trainTargets,y);
    accuracyVals(n,1) = 100 - c*100;
    
    [c,cm,ind,per] = confusion(testTargets,y);
    accuracyVals(n,2) = 100 - c*100;
end
accuracyVals
break

%%
testTimes = [];
for i = 1 : numCols
    tic
    testClass = net(x(:,i));
    testTimes = [testTimes toc];
end
mean(testTimes) * 1000

%%
for i = 1 : 225
    tic;
    net(x);
    testTimes = [testTimes toc];
end
mean(testTimes) * 1000