function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with 
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%

%h_theta=sigmoid(X*theta);
%Regularization_part=(lambda/(2*m))*(theta(2).^2 + theta(3).^2 +theta(4).^2);

%J=(-1/m)*sum((y.*log(h_theta)+(1-y).*log(1-h_theta))) + Regularization_part;

%grad(1)= sum((1/m).*(h_theta - y).*X(:,1));
%grad(2)= sum((1/m).*(h_theta - y).*X(:,2)) + (lambda/m)*theta(2);
%grad(3)= sum((1/m).*(h_theta - y).*X(:,3)) + (lambda/m)*theta(3);
%grad(4)= sum((1/m).*(h_theta - y).*X(:,4)) + (lambda/m)*theta(4);

h_theta=sigmoid(X*theta);
Regularization_part=(lambda/(2*m))*sum(theta(2:length(theta)).^2);
% X(a,b): all the elements in the vector from index a to b
% above theta(2: length(theta))
% where length(theta)=4, therefore, theta(1) is ommitted in this
% calculation



J = (-(1 / m) * sum(y.*log(h_theta) + (1-y).*log( 1 - h_theta)) ) + Regularization_part ;

grad = (1 / m) * sum( X .* repmat((h_theta - y), 1, size(X,2)));

% repmat(A,a,b)
% compies of matrix A , a times in row and b times in column

grad(:,2:length(grad)) = grad(:,2:length(grad)) + (lambda/m).*theta(2:length(theta))';



% =============================================================

grad = grad(:);

end
