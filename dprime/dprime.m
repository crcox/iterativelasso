function [dp,counts] = dprime(truth,prediction,jitter)
% Takes two binary vectors as input, one this represents the "truth", and
% the other the prediction of some classifier.
if nargin == 2
    jitter = 0.0005;
end
T = truth > 0;
P = prediction > 0;

hit = sum(bsxfun(@and,T,P));
fa = sum(bsxfun(@and,~T,P));

hitr = hit./sum(T);
far = fa./sum(~T);
if isnan(hitr)
    error('There are no targets in the test set.')
elseif isnan(far)
    error('The test set is entirely targets.')
end

hitr_capped = min(hitr,1-jitter);
hitr_capped = max(hitr_capped,jitter);

far_capped = min(far,1-jitter);
far_capped = max(far_capped,jitter);

dp = norminv(hitr_capped) - norminv(far_capped);

counts.targets = sum(T);
counts.distractors = sum(~T);
counts.hits = hit;
counts.hitrate = hitr;
counts.falsealarms = fa;
counts.falsealarmrate = far;
