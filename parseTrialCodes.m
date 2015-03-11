function metadata = parseTrialCodes(trialcodes,TRoffset)
  if nargin == 2
    ix = find(trialcodes) + TRoffset;
    if max(ix) > length(trialcodes)
      error('index exceeds number of TRs in data.')
    end
    metadata.selectedTRs = ix;
    trialcodes(trialcodes==0) = [];
  end

  n = length(trialcodes);
  metadata.True3D = false(n,1);
  metadata.TrueFast = false(n,1);
  metadata.TrueLeft = false(n,1);
  metadata.TrueAway = false(n,1);

  a = 1;
  b = n;

  trial_vector = trialcodes;

  temp = any(bsxfun(@eq,trial_vector, [25, 26, 41, 42]),2);
  metadata.True3D(a:b) = temp;

  temp = any(bsxfun(@eq,trial_vector, [37, 38, 41, 42]),2);
  metadata.TrueFast(a:b) = temp;

  temp = any(bsxfun(@eq,trial_vector, [21,37]),2);
  metadata.TrueLeft(a:b) = temp;

  temp = any(bsxfun(@eq,trial_vector, [26, 42]),2);
  metadata.TrueAway(a:b) = temp;
end

