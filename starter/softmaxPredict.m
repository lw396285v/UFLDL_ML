function [pred] = softmaxPredict(softmaxModel, data)

% softmaxModel - model trained using softmaxTrain
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set
%
% Your code should produce the prediction matrix 
% pred, where pred(i) is argmax_c P(y(c) | x(i)).
 
% Unroll the parameters from theta
theta = softmaxModel.optTheta;  % this provides a numClasses x inputSize matrix
pred = zeros(1, size(data, 2));

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute pred using theta assuming that the labels start 
%                from 1.
M = theta*data;
M = bsxfun(@minus, M, max(M, [], 1));
h = exp(M);
h = h./repmat(sum(h, 1),size(h,1),1);
max_h = max(h,[],1);
for j = 1:size(h,2)
    for i = 1:size(h,1)
        if h(i,j) == max_h(j)
           pred(j) = i;
        end
    end
end
% ---------------------------------------------------------------------

end

