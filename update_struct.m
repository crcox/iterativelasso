function x = update_struct(x,y)
  fn = fieldnames(y);
  val = struct2cell(y);
  n = length(fn);
  for i = 1:n
    x = setfield(x,fn{i},val{i});
  end
end
