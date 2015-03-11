function metadata = setup_cvblocks(ntrial,nMRIrun)
  metadata.mriRuns = reshape(repmat(1:nMRIrun,ntrial/nMRIrun,1),ntrial,1);
  metadata.CVBLOCKS = bsxfun(@eq, metadata.mriRuns, (1:nMRIrun));
	[~,cv] = ndgrid(1:(ntrial/nMRIrun),1:nMRIrun);
	metadata.cvind = cv(:);
end
