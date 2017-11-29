function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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

h_theta=sigmoid(X*theta);
Regularization_part=(lambda/(2*m))*(theta(2).^2 + theta(3).^2);

J=(-1/m)*sum((y.*log(h_theta)+(1-y).*log(1-h_theta))) + Regularization_part;

grad(1)= sum((1/m).*(h_theta - y).*X(:,1));
grad(2)= sum((1/m).*(h_theta - y).*X(:,2)) + (lambda/m)*theta(2);
grad(3)= sum((1/m).*(h_theta - y).*X(:,3)) + (lambda/m)*theta(3);





% =============================================================

end
