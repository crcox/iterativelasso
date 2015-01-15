function iterativelasso(X,Y,CVBLOCKS,varargin)
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
  CheckPoint = fullfile(ExpDir,'CHECKPOINT.mat');
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
		tmp = sprintf('iter%02d',iterCounter-1);
    iterDir = fullfile(ExpDir,tmp);
		if ~exist(iterDir,'dir')
			mkdir(iterDir)
		end

		% Setup a loop over holdout sets.
		cpb.start()
		while cc <= N_CV
			OMIT = cc;
			cvDir = sprintf('cv%02d', cc);
			outdir = fullfile(iterDir,cvDir);
			if ~exist(outdir,'dir')
				mkdir(outdir);
			end
			outdir_tuning = fullfile(iterDir,cvDir,'tuning');
			if ~exist(outdir_tuning,'dir');
				mkdir(outdir_tuning);
			end

      if cc > 1
        dpstr = sprintf('% 6.2f',cell2mat({fitObj.dp}));
        nnzstr = sprintf('% 6d',cell2mat({fitObj.df}));
      else
        dpstr = '';
        nnzstr = '';
      end

			text = sprintf('Progress: %d/%d\n  dp: %s\n nnz: %s\n', cc-1, N_CV, dpstr,nnzstr);
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
			tuneObj = cvglmnet(Xtrain_unused,Ytrain, ...
														 'binomial',opts_cv,'class',N_CV-1,fold_id);
			tuneObj = computeModelFit(tuneObj,X_unused);
			writeResults(outdir_tuning, tuneObj, uuv);

			% Set that lambda in the opts structure, and fit a new model.
			opts.lambda = tuneObj.lambda_min;
			tmpObj = glmnet(Xtrain_unused,Ytrain,'binomial',opts);
			tmpObj = computeModelFit(tmpObj,X_unused);
			fitObj(cc) = evaluateModelFit(tmpObj,Y,FINAL_HOLDOUT);
			writeResults(outdir, fitObj(cc), uuv);

			% Indicate which voxels were used/update the set of unused voxels.
			if cc == 1
				UNUSED_VOXELS{iterCounter+1} = UNUSED_VOXELS{iterCounter};
			end
			UNUSED_VOXELS{iterCounter+1}(uuv,cc) = fitObj(cc).beta==0;

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
		if nsCounter > STOP_CRIT;
			CRIT_REACHED = true(1);
			break
		end
		fprintf('\n');
		cc = 1;
		iterCounter = iterCounter + 1;
	end
	N_ITER = iterCounter;
  Notes = fullfile(ExpDir,'summary.json');
  savejson('',struct('niter',N_ITER,'ncv',N_CV),Notes);
% 	%% Write out data in a useful way.
%
%   N_LAMBDA = opts_cv.nlambda;
% 	BETA = cell(1,N_CV);
% 	[BETA{:}] = deal(zeros(N_VOX,N_ITER));
%   perfByLambda = zeros(N_VOX*N_ITER,5);
% 	for cc = 1:N_CV
% 		cvDir = sprintf('cv%02d',cc);
% 		for ii = 1:N_ITER
%       a = sub2ind([N_LAMBDA, N_ITER, N_CV], 1, ii, cc);
%       b = sub2ind([N_LAMBDA, N_ITER, N_CV], N_LAMBDA, ii, cc);
% 			iterDir = sprintf('iter%02d',N_ITER-1);
% 			dataPath = fullfile(ExpDir,iterDir,cvDir,'fitObj.mat');
% 			dataPath_cv = fullfile(ExpDir,iterDir,cvDir,'fitObj_cv.mat');
% 			z = ~UNUSED_VOXELS(:,cc,ii);
% 			load(dataPath,'beta');
% 			load(dataPath_cv,'cvm','cvsd','df','lambda');
% 			BETA{cc}(z,ii) = beta;
%       perfByLambda(a:b,:) = [cc,lambda,df,cvm,cvsd];
% 		end
% 		csvwrite(sprintf('betas_cv%02d.csv', cc), BETA{cc},'precision','%.6f');
% 	end
%   csvwrite('auc_by_lambda.csv', perfByLambda);

	%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
	disp('Fit Final Model')
	USEFUL_VOXELS = selectUsefulVoxels(UNUSED_VOXELS);

	err_ridge = zeros(1,N_CV);
	dp_ridge = zeros(1,N_CV);
  finalDir = fullfile(ExpDir,'final');

  if ~exist(finalDir,'dir')
    mkdir(finalDir);
  end

	for cc = 1:N_CV
    cvDir = sprintf('cv%02d', cc);
    outdir = fullfile(finalDir,cvDir);
		if ~exist(outdir,'dir')
			mkdir(outdir);
		end

		outdir_tuning = fullfile(finalDir,cvDir,'tuning');
		if ~exist(outdir_tuning,'dir')
			mkdir(outdir_tuning);
		end
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
		tuneObj = cvglmnet(Xtrain(:,uv),Ytrain, ...
														 'binomial',opts_final_cv,'class',N_CV-1,fold_id);
    tuneObj = computeModelFit(tuneObj,X(:,uv));
    writeResults(outdir_tuning, tuneObj, uv);

		% Set that lambda in the opts structure, and fit a new model.
		opts_final.lambda = tuneObj.lambda_min;
    tmpObj = glmnet(Xtrain(:,uv),Ytrain,'binomial',opts_final);
    tmpObj = computeModelFit(tmpObj,X(:,uv));
    fitObj(cc) = evaluateModelFit(tmpObj,Y,FINAL_HOLDOUT);
    writeResults(outdir, fitObj(cc), uv);
	end

	%% Write out data in a useful way. THIS IS WRONG@!!!!
%  N_LAMBDA = opts_final_cv.nlambda;
%	BETA = cell(1,N_CV);
%	[BETA{:}] = deal(zeros(N_VOX,1));
%  perfByLambda = zeros(N_VOX*N_LAMBDA,5);
%	for cc = 1:N_CV
%		cvDir = sprintf('cv%02d',cc);
%    a = sub2ind([N_LAMBDA, N_CV], 1, cc);
%    b = sub2ind([N_LAMBDA, N_CV], N_LAMBDA, cc);
%    dataPath = fullfile(ExpDir,finalDir,cvDir,'fitObj.mat');
%    dataPath_cv = fullfile(ExpDir,finalDir,cvDir,'fitObj_cv.mat');
%    z = ~UNUSED_VOXELS(:,cc,ii);
%    load(dataPath,'beta');
%    load(dataPath_cv,'cvm','cvsd','df','lambda');
%    BETA{cc}(z,ii) = beta;
%    perfByLambda(a:b,:) = [cc,lambda,df,cvm,cvsd];
%		csvwrite(sprintf('betas_cv%02d.csv', cc), BETA{cc},'precision','%.6f');
%	end
%  csvwrite('auc_by_lambda.csv', perfByLambda);

  %% Package results
%	results.errU = err_ridge;
%	results.dpU = dp_ridge;
%	results.UNUSED_VOXELS = UNUSED_VOXELS;
%	results.fitObj_ridge = fitObj_ridge;
%	results.fitObj_cv_ridge = fitObj_ridge_cv;
%	results.fitObj = fitObj(1:iterCounter,:);
%	results.fitObj_cv = fitObj_cv(1:iterCounter,:);
%	results.err = err(1:iterCounter,:);
%	results.dp = dp(1:iterCounter,:);
%	results.errU = err_ridge;
%	results.dpU = dp_ridge;
%	results.UNUSED_VOXELS = UNUSED_VOXELS(:,:,1:iterCounter);
	delete('CHECKPOINT.mat');

	%% Nested Functions
	function saveCheckpoint()
		save('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp','fitObj');
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

	function [cc, UNUSED_VOXELS, iterCounter, nsCounter, err, dp, fitObj] = loadCheckpoint() %#ok<STOUT>
	% Function will check for a checkpoint file and load its contents if it
	% is present. If a checkpoint file is not present, variables are
	% initiallized to appropriate starting values.
		load('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp','fitObj');
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
% 			dp = {fitObj.dp};
% 			if ~iscol(dp);
% 				dp = dp';
% 			end
% 			dpmat = cell2mat(dp);
%
% 			dpt = fitObj.dpt;
% 			if ~iscol(dpt)
% 				dpt = dpt'
% 			end
% 			dptmat = cell2mat(dpt);
%
% 			writeBinMatrix(fullfile(outdir,'dprime_test.bin'), dpmat);
% 			writeBinMatrix(fullfile(outdir,'dprime_train.bin'), dptmat);
% 			cvind = 1:N_CV;
% 			[cv,lam,subj,train,omit,alpha] = ndgrid(cvind,fitObj.lambda,1:nSubjects,[1,0],OMIT,ALPHA);
% 			dpvec = [cell2mat(cellfun(@(x) x(:), fitObj.dp,'unif',false)); ...
% 							 cell2mat(cellfun(@(x) x(:), fitObj.dpt,'unif',false))];
% 			dptbl = [cv(:),lam(:),subj(:),train(:),omit(:),alpha(:),dpvec];
% 			dptbl_header = {'cv','lam','subj','test','omit'};
% 			save(fullfile(outdir,'dprime_table.mat'),'dptbl','dptbl_header');
% 			csvwrite(fullfile(outdir,'dprime_table.csv'),dptbl);
% 			writeBinTable(fullfile(outdir,'dprime_table.bin'),dptbl,'compress',false);
% 		end
    % NB Cannot compress on CONDOR since that would require java.
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
  fwrite(f,size(tbl), 'uint', 'ieee-le');    % 4 byte, dims
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
