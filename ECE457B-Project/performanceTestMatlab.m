testTimes = [];
for i = 1 : numCols
    tic
    testClass = net(x(:,i));
    testTimes = [testTimes toc];
end
mean(testTimes) * 1000