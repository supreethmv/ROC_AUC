function p_value=DoPermutationTest(Xtrain,Ytrain,N)
%  A permutation test for MATLAB, nonparametric
%  method for testing whether two distributions are the same formulas taken
%  from Hein.pdf page 111,134


classes = unique(Ytrain);



% Fisher Score for the original data
tOrg=FScore(Xtrain,Ytrain,classes);



% Fisher Score for the permuted data
T_j = zeros(N,1);
for iter=1:N
    T_j(iter)=FScore(Xtrain,Ytrain(randperm(length(Ytrain))),classes);
end



% the p-value is the probability, given that the null hypothesis is true,
% of observing a value of the test statistic larger or equal than the one
% that has been observed,

p_value = 1/N*sum(T_j > tOrg);





function FS=FScore(Xtrain,Ytrain,classes)
% The most simple and thus often suboptimal method is to compute a score
% for each feature (based only on data available for this feature).

% Choosing the features with the Fisher score is optimal if the
% class-conditional distribution is Gaussian, where the individual features
% are conditionally independent and the variances of one feature being
% equal for both classes
% F = (m+ - m-)^2 / sigma+^2 + sigma-^2



p=Xtrain(Ytrain==classes(1));
n=Xtrain(Ytrain==classes(2));


p_mean = mean(p);
p_sigma = mean(p.^2)-p_mean;
n_mean = mean(n);
n_sigma = mean(n.^2)-n_mean;


FS=(p_mean-n_mean)^2/(p_sigma^2+n_sigma^2);