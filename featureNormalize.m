function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

m=size(X,1);
n=size(X,2);

fprintf('m inside featureNormalize %f\n',m);
fprintf('n inside featureNormalize %f\n',n);


for i=1:n
    mu(i)=mean(X(:,i));
    sigma(i)=std(X(:,i));
end

[mean1, mean2]=size(mu);
[deviation1, deviation2]=size(sigma);

fprintf('Mean size %f, %f\n',mean1, mean2);
fprintf('Mean(1) Mean(2) %f, %f \n',mu(1),mu(2));
fprintf('deviation size %f, %f\n',deviation1, deviation2);
fprintf('deviation(1) deviation(2) %f, %f \n',sigma(1),sigma(2));

    
for j=1:m
    for k=1:n
        X_norm(j,k)= (X(j,k)-mu(k))/sigma(k);
        %X_norm(j,k)=X_norm(j,k)/sigma(k);
        
    end
end

[X_norm1, X_norm2]=size(X_norm);
fprintf('X_norm %f, %f\n',X_norm1, X_norm2);


    






% ============================================================

end
