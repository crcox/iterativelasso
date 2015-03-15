function [finalModel,iterModels,finalTune,iterTune] = iterativelasso(X,Y,CVBLOCKS,varargin)
	defaultMaxIter = Inf;

	p = inputParser;
	addRequired(p, 'X');
	addRequired(p, 'Y');
	addRequired(p, 'CVBLOCKS');
	addParameter(p, 'MaxIter', defaultMaxIter);
	addParameter(p, 'ExpDir', '');
	parse(p,X,Y,CVBLOCKS,varargin{:});

	X = p.Results.X;
	Y = p.Results.Y;
	CVBLOCKS = p.Results.CVBLOCKS;
	MaxIter = p.Results.MaxIter;
	ExpDir = p.Results.ExpDir;

	N_CV = size(CVBLOCKS,2);
	N_VOX = size(X,2);

	STOP_CRIT = 2; % Number of consecutive non-significant iterations before
	               % breaking the loop.
	CRIT_REACHED = false;

	cpb = setupProgressBar(0,N_CV);
	CheckPoint = fullfile('CHECKPOINT.mat');
	if exist(CheckPoint,'file')
		[cc,UNUSED_VOXELS,iterCounter,nsCounter,err,dp,fitObj] = loadCheckpoint();
	else
		[cc,UNUSED_VOXELS,iterCounter,nsCounter,err,dp] = initializeWorkspace();
	end

	opts = glmnetSet();
	opts_cv = glmnetSet();
	opts_final = glmnetSet(struct('alpha',0));
	opts_final_cv = glmnetSet(struct('alpha',0));

	% Begin Iterative Lasso
	while iterCounter < MaxIter

		% Setup a loop over holdout sets.
		cpb.start()
		while cc <= N_CV
			OMIT = cc;
			text = sprintf('Progress: %d/%d\n', cc-1, N_CV);
			cpb.setValue(cc-1);
			cpb.setText(text);

			% Pick a final holdout set
			FINAL_HOLDOUT = CVBLOCKS(:,OMIT);

			% Remove the holdout set
			CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=OMIT);
			Xtrain = X(~FINAL_HOLDOUT,:);
			Ytrain = Y(~FINAL_HOLDOUT);

			% Convert CV2 to fold_id (for cvglmnet)
			fold_id = sum(bsxfun(@times,double(CV2),1:(N_CV-1)),2);

			% For some reason, this must be a row vector.
			fold_id = transpose(fold_id);

			% Run cvglmnet to determine a good lambda.
			uuv = UNUSED_VOXELS{iterCounter}(:,cc);
			X_unused = X(:,uuv);
			Xtrain_unused = Xtrain(:,uuv);
			if size(Xtrain_unused,2) > 0
				tuneObj = cvglmnet(Xtrain_unused,Ytrain, ...
									 'binomial',opts_cv,'class',N_CV-1,fold_id);
			else
				cc = cc + 1;
				saveCheckpoint();
				continue
			end

			tuneObj.mask = uuv;
			tuneObj.y = Y>0;
			tuneObj.testset = FINAL_HOLDOUT;
			tuneObj = computeModelFit(tuneObj,X_unused);
			tuneObj = rmfield(tuneObj,'glmnet_fit');
%			writeResults(outdir_tuning, tuneObj, uuv);

			% Set that lambda in the opts structure, and fit a new model.
			opts.lambda = tuneObj.lambda_min;
			tmpObj = glmnet(Xtrain_unused,Ytrain,'binomial',opts);
			tmpObj.mask = uuv;
			tmpObj.y = Y>0;
			tmpObj.testset = FINAL_HOLDOUT;
			tmpObj = computeModelFit(tmpObj,X_unused);
			fitObj(cc) = evaluateModelFit(tmpObj,Y,FINAL_HOLDOUT);
%			writeResults(outdir, fitObj(cc), uuv);

			% Indicate which voxels were used/update the set of unused voxels.
			if cc == 1
				UNUSED_VOXELS{iterCounter+1} = UNUSED_VOXELS{iterCounter};
			end
			UNUSED_VOXELS{iterCounter+1}(uuv,cc) = fitObj(cc).beta==0;

			% Log models for output (preallocating these things is more work than
			% it is worth... )
			iterModels(iterCounter,cc) = fitObj(cc); %#ok<AGROW>
			iterTune(iterCounter,cc) = tuneObj; %#ok<AGROW>

			% Save a checkpoint file
			cc = cc + 1;
			saveCheckpoint();
		end
		text = sprintf('Progress: %d/%d', cc-1, N_CV);
		cpb.setValue(cc-1);
		cpb.setText(text);
		cpb.stop();

		%% Test if dprime is significantly greater than zero.
		dp = cell2mat({fitObj.dp});
		disp(dp)
		h = ttest(dp, 0,'Alpha',0.05,'Tail','right');
		if isnan(h)
			h = false;
		end
		if h==true
			% If it is, reset nsCounter ...
			nsCounter = 0;
		else
			% If it is not, increment nsCounter ...
			nsCounter = nsCounter + 1;
		end

		% If after STOP_CRIT consecutive non-significant iterations, break.
		if nsCounter >= STOP_CRIT;
			CRIT_REACHED = true(1);
			break
		end
		fprintf('\n');
		cc = 1;
		iterCounter = iterCounter + 1;
	end
	N_ITER = iterCounter;

	% Save a summary json file.
	Notes = fullfile(ExpDir,'summary.json');
	savejson('',struct('niter',N_ITER,'ncv',N_CV),Notes);

	%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
	disp('Fit Final Model')
	USEFUL_VOXELS = selectUsefulVoxels(UNUSED_VOXELS);

	if any(USEFUL_VOXELS)
		err_ridge = zeros(1,N_CV);
		dp_ridge = zeros(1,N_CV);

		for cc = 1:N_CV
			disp(cc)

			% Remove the holdout set
			FINAL_HOLDOUT = CVBLOCKS(:,cc);
			CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc);
			Xtrain = X(~FINAL_HOLDOUT,:);
			Ytrain = Y(~FINAL_HOLDOUT);
			uv = USEFUL_VOXELS(:,cc);

			% Convert CV2 to fold_id
			fold_id = sum(bsxfun(@times,double(CV2),1:N_CV-1),2);

			% For some reason, this must be a row vector.
			fold_id = transpose(fold_id);

			% Run cvglmnet to determine a good lambda.
			tmpObj = cvglmnet(Xtrain(:,uv),Ytrain, ...
                               'binomial',opts_final_cv,'class',N_CV-1,fold_id);
			tmpObj.mask = uv;
			tmpObj = computeModelFit(tmpObj,X(:,uv));
			finalTune(1,cc) = rmfield(tmpObj,'glmnet_fit');
	%		writeResults(outdir_tuning, finalTune(cc), uv);

			% Set that lambda in the opts structure, and fit a new model.
			opts_final.lambda = finalTune(cc).lambda_min;
			tmpObj = glmnet(Xtrain(:,uv),Ytrain,'binomial',opts_final);
			tmpObj.mask = uv;
			tmpObj.y = Y>0;
			tmpObj.testset = FINAL_HOLDOUT;
			tmpObj = computeModelFit(tmpObj,X(:,uv));
			finalModel(1,cc) = evaluateModelFit(tmpObj,Y,FINAL_HOLDOUT);
	%		writeResults(outdir, finalModel(cc), uv);
		end
	else
		fprintf('No significant iterations. Cannot fit final model.\n');
		iterModels;
		iterTune;
		finalModel = struct();
		finalTune = struct();
	end

	delete('CHECKPOINT.mat');

	%% Nested Functions
	function saveCheckpoint()
		save('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp','iterModels');
	end

	function [cc, UNUSED_VOXELS, iterCounter, nsCounter, err, dp] = initializeWorkspace()
		fprintf('starting from scratch\n');
		iterCounter = 1;
		nsCounter = 0;
		cc = 1;
		dp = {zeros(1,N_CV)};
		err = {zeros(1,N_CV)};
		UNUSED_VOXELS = {true(N_VOX,N_CV)};
	end

	function [cc, UNUSED_VOXELS, iterCounter, nsCounter, err, dp, iterModels] = loadCheckpoint() %#ok<STOUT>
	% Function will check for a checkpoint file and load its contents if it
	% is present. If a checkpoint file is not present, variables are
	% initiallized to appropriate starting values.
		load('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp','iterModels');
		if cc < N_CV
			fprintf('++Resuming from CV%02d\n',cc);
		end
	end

	function fitObj = computeModelFit(fitObj,X)
		n = size(X,1);
		if strcmp(fitObj.class,'cv.glmnet')
			B = [fitObj.glmnet_fit.a0;fitObj.glmnet_fit.beta];
		else
			B = [fitObj.a0;fitObj.beta];
		end
		fitObj.Yh = [ones(n,1),X] * B;
	end

	function fitObj = evaluateModelFit(fitObj,y,cv)
		fitObj.dp = dprime(y(cv),fitObj.Yh(cv)>0);
		fitObj.dpt = dprime(y(~cv),fitObj.Yh(~cv)>0);
	end

	function x = selectUsefulVoxels(USED_VOXELS)
		if N_ITER == MaxIter
			x = ~USED_VOXELS{end};
		elseif CRIT_REACHED
			x = ~USED_VOXELS{end-STOP_CRIT};
		end
	end

	function writeResults(outdir,fitObj,mask)
		save(fullfile(outdir,'fitObj.mat'),'-struct','fitObj');
		betas = zeros(length(mask),length(fitObj.lambda));
		writeBinMatrix(fullfile(outdir,'voxel_mask.bin'), mask);
		writeBinMatrix(fullfile(outdir,'fittedVals.bin'), full(fitObj.Yh));
		if strcmp(fitObj.class,'cv.glmnet')
			writeBinMatrix(fullfile(outdir,'auc_tuning.bin'), full(fitObj.cvm));
			betas(mask,:) = full(fitObj.glmnet_fit.beta);
		else
			betas(mask) = full(fitObj.beta);
		end
		writeBinMatrix(fullfile(outdir,'beta.bin'), betas);
	end
end

function writeBinMatrix(filename,X)
% Column-major order.
	f = fopen(fullfile(filename),'w');
	fwrite(f,size(X), 'int', 'ieee-le'); % 4 byte
	fwrite(f, X, 'double', 'ieee-le');   % 8 byte
	fclose(f);
end

function writeBinTable(filename,tbl,varargin)
% Column-major order
% Cannot compress on CONDOR since that would require java.
	p = inputParser;
	p.addOptional('compress',false,@islogical);
	parse(p, varargin{:});
	f = fopen(filename,'w');
	fwrite(f,size(tbl), 'uint', 'ieee-le');	  % 4 byte, dims
	fwrite(f, tbl(:,1), 'uint8', 'ieee-le');   % 1 byte, cv
	fwrite(f, tbl(:,2), 'float', 'ieee-le');   % 4 byte, lam
	fwrite(f, tbl(:,3), 'uint8', 'ieee-le');   % 1 byte, subj
	fwrite(f, tbl(:,4), 'uint8', 'ieee-le');   % 1 byte, isTest
	fwrite(f, tbl(:,5), 'uint8', 'ieee-le');   % 1 byte, omit
	fwrite(f, tbl(:,6), 'float', 'ieee-le');   % 4 byte, alpha
	fwrite(f, tbl(:,7), 'float', 'ieee-le');   % 4 byte, dprime
	fclose(f);
	if p.Results.compress == true
		gzip(filename);
		delete(filename)
	end
end

%% Private Functions
function cpb = setupProgressBar(minval,maxval)
	cpb = ConsoleProgressBar();

	% Set progress bar parameters
	cpb.setLeftMargin(4);   % progress bar left margin
	cpb.setTopMargin(1);    % rows margin

	cpb.setLength(40);      % progress bar length: [.....]
	cpb.setMinimum(minval); % minimum value of progress range [min max]
	cpb.setMaximum(maxval); % maximum value of progress range [min max]

	% Set text position
	cpb.setPercentPosition('left');
	cpb.setTextPosition('right');

% 	cpb.setElapsedTimeVisible(1);
% 	cpb.setRemainedTimeVisible(1);
%
% 	cpb.setElapsedTimePosition('left');
% 	cpb.setRemainedTimePosition('right');
end
