function [confusionM, classPriors] = confusionMatrix(labels,decisions, l, d)
% assumes labels are in {1,...,L} and decisions are in {1,...,D}
L = l;
D = d;
confusionM = zeros(D,L);
for l = 1:L
    Nl = length(find(labels==l));
    for d = 1:D
        Ndl = length(find(labels==l & decisions==d));
        confusionM(d,l) = Ndl/Nl;
        
    end
    confusionM(isnan(confusionM)) = 0;
    classPriors(l,1) = Nl/length(labels); % class prior for label l
end
if L==D
    pCorrect = sum(diag(confusionM).*classPriors);
    pError = 1-pCorrect;
end
%expectedRisk = sum(sum(lossMatrix.*confusionMatrix.*repmat(classPriors,1,L),2),1);