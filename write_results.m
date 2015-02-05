function write_results(outdir,fitObj,tuneObj)
  % USAGE: outdir is the name of the root of the output directory that should
  % be composed from in data in fitObj. The tuneObj is optional, but if it is
  % supplied then the tuning data will be nested within the folder of the model
  % it pertains to.
  if nargin < 3
    writeTuningData = false;
  else
    writeTuningData = true;
  end

  niter = size(fitObj,1);
  ncv = size(fitObj,2);
  for ii = 1:niter
    if niter > 1
      iterDir = sprintf('iter%02d',ii-1);
      curDir = fullfile(outdir,iterDir);
      if ~exist(curDir,'dir')
        mkdir(curDir)
      end
    else
      curDir = outdir;
    end

    for cc = 1:ncv
      cvDirName = sprintf('cv%02d', cc);
      cvDir = fullfile(curDir,cvDirName);
      if ~exist(cvDir,'dir')
        mkdir(cvDir);
      end
      write_model(cvDir, fitObj(ii,cc))

      if writeTuningData
        tuneDir = fullfile(cvDir,'tuning');
        if ~exist(tuneDir,'dir')
          mkdir(tuneDir);
        end
        write_model(tuneDir, tuneObj(ii,cc))
      end
    end
  end
end

%% Private Functions
function write_model(outdir,fitObj)
  save(fullfile(outdir,'fitObj.mat'),'-struct','fitObj');
  betas = zeros(length(fitObj.mask),length(fitObj.lambda));
  writeBinMatrix(fullfile(outdir,'voxel_mask.bin'), fitObj.mask);
  writeBinMatrix(fullfile(outdir,'fittedVals.bin'), full(fitObj.Yh));
  if strcmp(fitObj.class,'cv.glmnet')
    writeBinMatrix(fullfile(outdir,'auc_tuning.bin'), full(fitObj.cvm));
    % betas(fitObj.mask,:) = full(fitObj.glmnet_fit.beta); % This data is not being tracked anymore for memory reasons.
  else
    betas(fitObj.mask) = full(fitObj.beta);
    csvwrite(fullfile(outdir,'itemscore.csv'),[fitObj.y,fitObj.testset,fitObj.Yh])
  end
  writeBinMatrix(fullfile(outdir,'beta.bin'), betas);
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
