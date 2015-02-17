function runIterativeLasso(jobdir)
	if nargin == 0
		params = loadjson('params.json')
		jobdir = params.expdir;
	else
		params = loadjson(fullfile(jobdir,'params.json'));
	end

	%% Load the data
	load(params.data, 'X','trialcodes','ijk');
	nMRIrun = params.nMRIrun;
	target = params.target;

	%% Generate metadata
	if isfield(params,'offsetFromStimOnset')
		metadata = parseTrialCodes(trialcodes,params.offsetFromStimOnset);
	else
		metadata = parseTrialCodes(trialcodes);
	end

	if isfield(metadata,'selectedTRs')
		X = X(metadata.selectedTRs,:);
	end

	[ntrial,~]=size(X);
	metadata = update_struct(metadata, setup_cvblocks(ntrial,nMRIrun));

	y = double(metadata.(target));
	cv = metadata.cvind;
	ncv = nMRIrun;
	CVB = metadata.CVBLOCKS;

	% zscore by run
	Xc = mat2cell(X,repmat(nMRIrun,size(X,1)/nMRIrun,1),size(X,2));
	X = cell2mat(cellfun(@zscore,Xc,'unif',0));

	%% Apply data reduction/filtering
	% Only 2d motion
	filter = true(size(X,1),1);
	if isfield(params,'Only2D') && params.Only2D
		filter = filter & ~metadata.True3D;
	end
	if isfield(params,'Only3D') && params.Only3D
		filter = filter & metadata.True3D;
	end
	% Only slow motion
	if isfield(params,'OnlySlow') && params.OnlySlow
		filter = filter & ~metadata.TrueFast;
	end

	X = X(filter,:);
	y = y(filter);
	cv = cv(filter);
	CVB = CVB(filter,:);

	% Remove outliers
	size(X)
	[X, retain] = removeOutliers(X);
	size(X)

	if ~all(retain.stimuli)
		y = y(retain.stimuli);
		cv = cv(retain.stimuli);
		CVB = CVB(retain.stimuli,:);
	end

	if ~all(retain.voxels)
		ijk = ijk(retain.voxels,:);
	end
	csvwrite(fullfile(jobdir,'ijk.csv'),ijk);

% 	X = zscore(X);
	%% Run Iterative Lasso
	[finalModel,iterModels,finalTune,iterTune] = iterativelasso(X,y,CVB,'ExpDir',jobdir);

	%% Save Results to Disk
	if ~isempty(fieldnames(finalModel))
		write_results(fullfile(jobdir,'final'),finalModel,finalTune);
	end
	write_results(fullfile(jobdir,'iterations'),iterModels,iterTune);
end
