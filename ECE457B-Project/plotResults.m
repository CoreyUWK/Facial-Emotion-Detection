close all

%% Plot Results
epoch
lastEpoch = min(epoch,1000);
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

% Plot 3: Confusion Plot for Training and Testing
figure(3);
plotconfusion(trainingTargets, trainingOutput, 'Training')

figure(4);
title('Test');
plotconfusion(testTargets, testOutput, 'Test')

% % Plot 4: ROC for Training and Testing
% figure(5);
% title('Training');
% plotroc(trainingTargets, trainingOutput, 'Train')
% 
% figure(6);
% title('Test');
% plotroc(testTargets, testOutput, 'Test')
