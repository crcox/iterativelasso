function [Aredux, reduxFilter]=removeOutliers(A)
% This function will z-score the row and column means and remove those
% which have a z-score greater than 5.  The process it performed on
% columns, and then rows.  Each time outliers are removed, the reduced data
% is checked again for outliers, until no outliers remain.

    Aredux=A;

    % For columns...
    outliers = abs(zscore(mean(A,1))) > 5;
    outCount = sum(outliers);
%     outliers = [];
    while outCount > 0
				z = outliers;
        outliers(~z) = abs(zscore(mean(A(:,~z),1))) > 5;
        outCount = sum(outliers(~z));
    end
    reduxFilter.voxels = ~outliers;
		Aredux(:,outliers) = [];

    % For rows...
    outliers = abs(zscore(mean(Aredux,2))) > 5;
    outliers = outliers'; % Mean by row creates a vector n-by-1 instead of 1-by-n.
    outCount = sum(outliers);
    while outCount > 0
				z = outliers;
				outliers(~z) = abs(zscore(mean(Aredux(~z,:),2))) > 5;
				outCount = sum(outliers(~z));
    end
    reduxFilter.stimuli = ~outliers;
		Aredux(outliers,:) = [];
end
