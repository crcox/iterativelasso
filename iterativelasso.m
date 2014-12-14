function iterative_lasso(X,Y,CVBLOCKS,varargin)
  defaultMaxIter = Inf;

  p = inputParser;
  addRequired(p, 'X');
  addRequired(p, 'Y');
  addRequired(p, 'CV');
  addParameter(p, 'MaxIter', defaultMaxIter);
  parse(p,X,Y,CV,varargin{:});

  X = p.Results.X;
  Y = p.Results.Y;
  CV = p.Results.CV;
  MaxIter = p.Results.MaxIter;

  N_CV = size(CVBLOCKS,2);
	N_VOX = size(X,2);
	N_ITER_EST = 100; % This is just used for matrix pre-allocation. Because
	                  % the number of iterations is unknown ahead of time in
										% each case, it is better to just preallocate too much
										% space and throw away whatever is unused later.
										% Dynamically growing things is always slower than pre-
										% allocation.
	STOP_CRIT = 2; % Number of consecutive non-significant iterations before
	               % breaking the loop.
	cpb = setupProgressBar(0,N_CV);
	[start_cc,UNUSED_VOXELS,iterCounter,nsCounter,err,dp] = loadCheckpoint();

	% Begin Iterative Lasso
	while iterCounter < MaxIter:
		iterCounter = iterCounter + 1;
		iterDir = sprintf('iter%02d',iterCounter-1);
		mkdir(iterDir)

		% Setup a loop over holdout sets.
		if start_cc <= N_CV
			cpb.start()

			for cc = start_cc:N_CV
				cvDir = sprintf('cv%02d', cc);
				mkdir(iterDir,cvDir);

				text = sprintf('Progress: %d/%d', cc, N_CV);
				cpb.setValue(cc);
				cpb.setText(text);

				% Select the voxels that have not been used look like.
				UNUSED_VOXELS(:,cc,iterCounter)

				% Pick a final holdout set
				FINAL_HOLDOUT = CVBLOCKS(:,cc);

				% Remove the holdout set
				CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc);
				Xtrain = X(~FINAL_HOLDOUT,:);
				Ytrain = Y(~FINAL_HOLDOUT);

				% Convert CV2 to fold_id (for cvglmnet)
				fold_id = sum(bsxfun(@times,double(CV2),1:(N_CV-1)),2);

				% For some reason, this must be a row vector.
				fold_id = transpose(fold_id);

				% Run cvglmnet to determine a good lambda.
				Xtrain_unused = Xtrain(:,UNUSED_VOXELS(:,iterCounter));
				fitObj_cv = cvglmnet(Xtrain_unused,Ytrain, ...
															 'binomial',opts,'auc',N_CV-1,fold_id);
				save(fullfile(iterDir,cvDir,'fitObj_cv.mat'), '-struct','fitObj_cv');

				% Set that lambda in the opts structure, and fit a new model.
				opts.lambda = fitObj_cv.lambda_min;
				fitObj = glmnet(Xtrain_unused,Ytrain,'binomial',opts);
				save(fullfile(iterDir,cvDir,'fitObj.mat'), '-struct','fitObj');

				% Unset lambda, so next time around cvglmnet will look for lambda
				% itself.
				opts = rmfield(opts,'lambda');

				% Evaluate this new model on the holdout set.
				% Step 1: compute the model predictions.
				X_unused = X(:,unVox);
				yhat = (X_unused * fitObj.beta)+fitObj.a0;

				% Step 2: compute the error of those predictions.
				err(cc, iterCounter) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));

				% Step 3: compute the sensitivity of those predictions (dprime).
				dp(cc, iterCounter) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);

				% Indicate which voxels were used/update the set of unused voxels.
				UNUSED_VOXELS(:,cc,iterCounter+1) = UNUSED_VOXELS(:,cc,iterCounter);
				UNUSED_VOXELS(UNUSED_VOXELS(:,cc,iterCounter),cc,iterCounter+1) = fitObj.beta==0;

				% Save a checkpoint file
				saveCheckpoint();
				end
			cpb.stop();
			start_cc = 1;
		end

		%% Test if the dprime is significantly greater than zero.
		h = ttest(dp(:,iterCounter),0,'Alpha',0.05,'Tail','right');
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
			break
		end
		fprintf('\n');
	end

	%% Write out data in a useful way.
	N_ITER = iterCounter;
  N_LAMBDA = opts.nlambda;
	BETA = cell(1,N_CV);
	[BETA{:}] = deal(zeros(N_VOX,N_ITER));
  perfByLambda = zeros(N_VOX*N_ITER*100,5);
	for cc = 1:N_CV
		cvDir = sprintf('cv%02d',cv);
		for ii = 1:N_ITER
      a = sub2ind([N_LAMBDA, N_ITER, N_CV], 1, ii, cc);
      b = sub2ind([N_LAMBDA, N_ITER, N_CV], N_LAMBDA, ii, cc);
			iterDir = sprintf('iter%02d',N_ITER-1);
			dataPath = fullfile(iterDir,cvDir,'fitObj.mat');
			dataPath_cv = fullfile(iterDir,cvDir,'fitObj_cv.mat');
			z = ~UNUSED_VOXELS(:,cc,ii);
			load(dataPath,'beta');
			load(dataPath_cv,'cvm','cvsd','df','lambda');
			BETA{cc}(z,ii) = beta;
      perfByLambda(a:b,:) = [cc,lambda,df,cvm,cvsd];
		end
		csvwrite(sprintf('betas_cv%02d.csv', cc), BETA{cc},'precision','%.6f');
	end
  csvwrite('auc_by_lambda.csv', perfByLambda);

	%% Fit model using all voxels from models with above chance performance (1:(ii-3)).
	disp('Fit Final Model')
	USEFUL_VOXELS = ~UNUSED_VOXELS(:,:,iterCounter-3);
	USEFUL_VOXELS = any(USEFUL_VOXELS,3);

	opts.alpha = 0; % ridge regression
% 	fitObj_ridge = init_glmnet_result_struct('glmnet',[1, ENV.N_CV]);
% 	fitObj_cv_ridge = init_glmnet_result_struct('cvglmnet',[1, ENV.N_CV]);

	err_ridge = zeros(1,N_CV);
	dp_ridge = zeros(1,N_CV);
	for cc = 1:N_CV
		disp(cc)
		% Remove the holdout set
		FINAL_HOLDOUT = CVBLOCKS(:,cc);
		CV2 = CVBLOCKS(~FINAL_HOLDOUT,(1:N_CV)~=cc);
		Xtrain = X(~FINAL_HOLDOUT,:);
		Ytrain = Y(~FINAL_HOLDOUT);

		% Convert CV2 to fold_id
		fold_id = sum(bsxfun(@times,double(CV2),1:N_CV-1),2);

		% For some reason, this must be a row vector.
		fold_id = transpose(fold_id);

		% Run cvglmnet to determine a good lambda.
		fitObj_ridge_cv = cvglmnet(Xtrain(:,USEFUL_VOXELS),Ytrain, ...
														 'binomial',opts,'class',N_CV-1,fold_id);
		save(fullfile(iterDir,cvDir,'fitObj_ridge_cv.mat'), 'fitObj_ridge_cv');

		% Set that lambda in the opts structure, and fit a new model.
		opts.lambda = fitObj_ridge_cv.lambda_min;
		fitObj_ridge = glmnet(Xtrain(:,USEFUL_VOXELS),Ytrain,'binomial',opts);
		save(fullfile(iterDir,cvDir,'fitObj_ridge.mat'), 'fitObj_ridge');

		% Unset lambda, so next time around cvglmnet will look for lambda
		% itself.
		opts = rmfield(opts,'lambda');

		% Evaluate this new model on the holdout set.
		% Step 1: compute the model predictions.
		yhat = (X(:,USEFUL_VOXELS)*fitObj_ridge(cc).beta)+fitObj(cc).a0;
		% Step 2: compute the error of those predictions.
		err_ridge(cc) = 1-mean(Y(FINAL_HOLDOUT)==(yhat(FINAL_HOLDOUT)>0));
		% Step 3: compute the sensitivity of those predictions (dprime).
		dp_ridge(cc) = dprimeCV(Y,yhat>0,FINAL_HOLDOUT);
	end

	%% Package results
	results.errU = err_ridge;
	results.dpU = dp_ridge;
	results.UNUSED_VOXELS = UNUSED_VOXELS;
	results.fitObj_ridge = fitObj_ridge;
	results.fitObj_cv_ridge = fitObj_ridge_cv;
	results.fitObj = fitObj(1:iterCounter,:);
	results.fitObj_cv = fitObj_cv(1:iterCounter,:);
	results.err = err(1:iterCounter,:);
	results.dp = dp(1:iterCounter,:);
	results.errU = err_ridge;
	results.dpU = dp_ridge;
	results.UNUSED_VOXELS = UNUSED_VOXELS(:,:,1:iterCounter);
	delete('CHECKPOINT.mat');

	%% Nested Functions
	function saveCheckpoint()
		save('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp');
	end

	function [cc, UNUSED_VOXELS, iterCounter, nsCounter, err, dp] = loadCheckpoint()
	% Function will check for a checkpoint file and load its contents if it
	% is present. If a checkpoint file is not present, variables are
	% initiallized to appropriate starting values.
		if exist(fullfile(pwd, 'CHECKPOINT.mat'), 'file') == 2
			load('CHECKPOINT.mat','cc','UNUSED_VOXELS','iterCounter','nsCounter','err','dp');
			cc = mod(cc+1,ENV.N_CV)+1; %#ok<NODEF>
			fprintf('++Resuming from CV%02d\n',cc);
		else
			fprintf('starting from scratch\n');
			iterCounter = 0;
			nsCounter = 0;
			cc = 1;
			dp = zeros(N_ITER_EST,N_CV);
			err = zeros(N_ITER_EST,N_CV);
			UNUSED_VOXELS = true(N_VOX,N_CV,N_ITER_EST);
		end
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
