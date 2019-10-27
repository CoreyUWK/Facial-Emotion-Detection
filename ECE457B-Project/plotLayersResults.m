clear;
close all;

load('Results-Layers.csv');

results = Results_Layers(2:end, 2:end);

training5 = results(:,1);
test5 = results(:,2);
training20 = results(:,3);
test20 = results(:,4);
training50 = results(:,5);
test50 = results(:,6);

layers = (1:3)';

figure(1);
plot(layers, training5, layers, training20, layers, training50);
title('Training Classification Results');
xlabel('Number of Layers');
ylabel('Correctly Classified (%)');

figure(2);
plot(layers, test5, layers, test20, layers, test50);
title('Test Classification Results');
xlabel('Number of Layers');
ylabel('Correctly Classified (%)');
legend('5 Neurons', '20 Neurons', '50 Neurons');