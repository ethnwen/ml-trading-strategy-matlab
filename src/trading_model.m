clear; clc;

data = readtable('../data/AAPL.csv');
prices = data.AdjClose;

returns = diff(prices)./prices(1:end-1);

ma5 = movmean(prices,5);
ma20 = movmean(prices,20);
momentum = prices - ma20;
volatility = movstd(returns,5);

X = [ma5(21:end), ma20(21:end), momentum(21:end), volatility(21:end)];
y = returns(20:end);

n = size(X,1);
split = floor(0.7*n);

Xtrain = X(1:split,:);
ytrain = y(1:split);

Xtest = X(split+1:end,:);
ytest = y(split+1:end);

model = fitrensemble(Xtrain, ytrain);
ypred = predict(model, Xtest);

signal = ypred > 0;
strategy_returns = signal .* ytest;

cum_market = cumsum(ytest);
cum_strategy = cumsum(strategy_returns);

figure;
plot(cum_market,'LineWidth',1.5); hold on;
plot(cum_strategy,'LineWidth',1.5);
legend('Buy & Hold','ML Strategy');
title('Cumulative Returns Comparison');

sharpe = mean(strategy_returns)/std(strategy_returns)*sqrt(252);
fprintf('Sharpe Ratio: %.2f\n', sharpe);
