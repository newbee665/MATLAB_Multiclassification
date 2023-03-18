function [RANKED, WEIGHT] = relief( X, Y, K )

fprintf('\n+ Feature selection method: Relief-F \n');
%% Wrapper: use Matlab implementation
[RANKED,WEIGHT] = relieff(X,Y,K);
% Matlab Code-Library for Feature Selection

