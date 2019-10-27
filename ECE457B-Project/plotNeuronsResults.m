clear;
close all;

load('Results-Neurons.csv');

results = Results_Neurons(1:12,:);

figure(1);
plot(results(:,1), results(:,2)*100, 'x-', results(:,1), results(:,3)*100,'x-', 'LineWidth',2);
title('Classification Results', 'FontSize', 15);
xlabel('Number of Neurons');
ylabel('Correctly Classified (%)');
ylim([50 102]);
legend('Training', 'Test', 'Location', 'SouthEast')

load('Results-Matlab-Neurons.csv');

results = Results_Matlab_Neurons(1:12,1:end);

figure(2);
plot(results(:,1), results(:,2), 'x-', results(:,1), results(:,3),'x-', 'LineWidth',2);
title('Classification Results', 'FontSize', 15);
xlabel('Number of Neurons');
ylabel('Correctly Classified (%)');
ylim([50 102]);
legend('Training', 'Test', 'Location', 'SouthEast')
