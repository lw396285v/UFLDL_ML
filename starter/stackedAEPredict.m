function [pred] = stackedAEPredict(theta, inputSize, hiddenSize, numClasses, netconfig, data)
                                         
% stackedAEPredict: Takes a trained theta and a test data set,
% and returns the predicted labels for each example.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 

% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
%% Unroll theta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.

numStack = numel(stack);
numCases = size(data,2);
z_cache = cell(size(stack));
a_cache = cell(size(stack));
z_cache{1} = stack{1}.w * data + stack{1}.b;
a_cache{1} = sigmoid(z_cache{1});
for i = 2:numStack
    z_cache{i} = stack{i}.w * a_cache{i-1} + stack{i}.b;
    a_cache{i} = sigmoid(z_cache{i});
end
M = softmaxTheta*a_cache{numStack};
M = bsxfun(@minus, M, max(M, [], 1));
softmax_predict = exp(M);
softmax_predict = softmax_predict./sum(softmax_predict,1);
max_pred = max(softmax_predict,[],1);
pred = zeros(1,numCases);
for i =1:numCases
    for j = 1:numClasses
        if softmax_predict(j,i) == max_pred(i)
           pred(i) = j; 
           break;
        end
    end
end
% -----------------------------------------------------------

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
