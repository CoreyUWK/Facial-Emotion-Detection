% To be run after training a Matlab neural network (matlabPatternNet)
% Calculates the average time required to classify an input

testTimes = [];
for i = 1 : numCols
    tic
    testClass = net(x(:,i));
    testTimes = [testTimes toc];
end

avgTimeMs = mean(testTimes) * 1000;
disp(['Average Time to Classify: ', num2str(avgTimeMs), ' ms']);