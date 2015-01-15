function results = main()
	%% Set Variables
  PARAMS = loadjson('PARAMS.json');
  if isfield(PARAMS, 'MaxIter')
    MaxIter = PARAMS['MaxIter'];
  else
    MaxIter = 0;
  end
	opts = glmnetSet();
	opts.alpha = 1; % 1 means LASSO; 0 means Ridge
	[Y, CVBLOCKS] = loadMetadata(PARAMS['metadata'], PARAMS['TargetCategory']);

  %% Start the loop
  for ss = 1:length(PARAMS.data)
		X = loadFuncData(PARAMS.data{ss});
		CVBLOCKS = CVBLOCKS{ss};
		Y = Y{ss};
    if MaxIter > 0
		  results = iterativelasso(X,Y,CVBLOCKS,'MaxIter',MaxIter);
    else
		  results = iterativelasso(X,Y,CVBLOCKS);
    end
	end
end

%% Private Functions
function [Y, CVBLOCKS] = loadMetadata(pathToMetadata,TargetCategory)
  load(pathToMetadata, 'metadata');
  Y = {metadata.(TargetCategory)};
  CVBLOCKS = {metadata.CVBLOCKS};
end

function X = loadFuncData(pathToFuncData)
   load(pathToFuncData,'X');
end
