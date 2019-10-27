% Arief Koesdwiady
% ECE457B Tutorial
% Description: This code visualizes 2-dimensional handwriting data, which are trained
% using Neural Network. The last part of the code shows the decision
% boundaries produced by the NN, which will classify whether the data
% belongs to class 1 (handwriting of number 0), 2 (handwriting of number 1), 3
% (handwriting of number 2)
close all 
clear 
clc

% STEP 1
% Load the data
load('Dataset.mat');
%S.allLandmarks
%S.outputEmotions

% Plot the data together with their images.
% The images of 0, 1, and 2 represent the labels (type of handwriting) of the data.
% figure(1)
% plotimages(images,features,0.01,0.8);
% hold on
% grid on
% 
% pause

% STEP 2
% Plot the data in another form, i.e., using markers.
% figure(2)
% plot(allLandmarks(:68,1),allLandmarks(69:136,1),'ms','MarkerEdgeColor','b','MarkerFaceColor','g','MarkerSize',11)
% hold on
% grid on
% plot(features(1,101:200),features(2,101:200),'bs','MarkerEdgeColor','r','MarkerFaceColor','b','MarkerSize',11)
% plot(features(1,201:300),features(2,201:300),'ro','MarkerEdgeColor','y','MarkerFaceColor','r','MarkerSize',11)
% 
% pause

% STEP 3
% Train a neural network
net = patternnet(3); % Number of neurons = 3;
net.divideParam.trainRatio = 125/225; % training set [%]
net.divideParam.valRatio = 25/225; % validation set [%]
net.divideParam.testRatio = 75/225; % test set [%]

net = train(net,allLandmarks,outputEmotions);

% show network
view(net)
pause

% STEP 4
% Create the decision boundaries
span = -0.15:.0005:0.25;
[P1,P2] = meshgrid(span,span);
pp = [P1(:) P2(:)]';

aa = net(pp);
aa = vec2ind(aa);

% Plot classification regions
figure(2)
mesh(P1,P2,reshape(aa,length(span),length(span))-4);
colormap gray

% use nnstart
% get weights
wb = getwb(Emotion_Detection_Network_MatLib)
% simulate 
outputs = sim(Emotion_Detection_Network_MatLib,allLandmarks(:,1))  

