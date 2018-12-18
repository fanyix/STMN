function value = load(str, varargin)
%LOAD Load matlab value from a JSON string.
%
%   value = json.load(str)
%   value = json.dump(..., optionName, optionValue, ...)
%
% The function parses a JSON string into a matlab value. By default,
% numeric literals are converted to double, string is converted to a char
% array, logical literals are converted to logical. A JSON array is converted
% to either a double array, a logical array, a cell array, or a struct
% array. A JSON object is converted to a struct array.
%
% OPTIONS
%
% The function takes following options.
%
%   'MergeCell'   Try to convert a JSON array into a double, a logical, or
%                 a struct array when possible. Default true.
%
%   'ColMajor'    Represent matrix in column-major order. Default false.
%
% EXAMPLE
%
%   >> value = json.load('{"char":"hello","matrix":[[1,3],[4,2]]}')
%   value =
%
%         char: 'hello'
%       matrix: [2x2 double]
%
%   >> value = json.load('[[1,2,3],[4,5,6]]')
%   value =
%
%        1     2     3
%        4     5     6
%
%   >> value = json.load('[[1,2,3],[4,5,6]]', 'ColMajor', true)
%   value =
%        1     4
%        2     5
%        3     6
%
%   >> value = json.load('[[1,2,3],[4,5,6]]', 'MergeCell', false)
%   value =
%
%       {1x3 cell}    {1x3 cell}
%
% NOTE
%
% Since any matlab values are an array, it is impossible to uniquely map
% all JSON primitives to matlab values. This implementation aims to have
% better interoperability across platforms. Therefore, some matlab values
% cannot be represented in a JSON string. For example, '[1,2,3]' is mapped to
% either [1, 2, 3] or {{1}, {2}, {3}} depending on 'MergeCell' option, but
% cannot produce {1, 2, 3}.
%
% See also json.dump json.read

  json.startup('WarnOnAddpath', true);
  options.MergeCell = true;
  options.ColMajor = false;
  options = getOptions(options, varargin{:});

  str = strtrim(str);
  assert(~isempty(str), 'Empty JSON string.');
  singleton = false;
  if str(1)=='{'
    node = org.json.JSONObject(java.lang.String(str));
  else
    singleton = str(1) ~= '[' && str(end) ~= ']';
    if singleton, str = ['[',str,']']; end
    node = org.json.JSONArray(java.lang.String(str));
  end
  value = parseData(node, options);
  if singleton
    value = value{:};
  end
end

function value = parseData(node, options)
%PARSEDATA
  if isa(node, 'char')
    value = char(node);
  elseif isa(node, 'double')
    value = double(node);
  elseif isa(node, 'logical')
    value = logical(node);
  elseif isa(node, 'org.json.JSONArray')
    value = cell(node.length() > 0, node.length());
    for i = 1:node.length()
      value{i} = parseData(node.get(i-1), options);
    end
    if options.MergeCell
      value = mergeCell(value, options);
    end
  elseif isa(node, 'org.json.JSONObject')
    value = struct;
    itr = node.keys();
    while itr.hasNext()
      key = itr.next();
      field = char(key);
      safe_field = genvarname(char(key), fieldnames(value));
      if ~strcmp(field, safe_field)
        warning('json:fieldNameConflict', ...
                'Field %s renamed to %s', field, safe_field);
      end
      value.(safe_field) = parseData(node.get(java.lang.String(key)), options);
    end
  elseif isa(node, 'org.json.JSONObject$Null')
    value = [];
  else
    error('json:typeError', 'Unknown data type: %s', class(node));
  end
end

function value = mergeCell(value, options)
%MERGECELL
  if isempty(value) || all(cellfun(@isempty, value))
    return;
  end
  if isscalar(value)
    return;
  end
  if ~all(cellfun(@isscalar, value)) && all(cellfun(@ischar, value))
    return;
  end

  if isMergeable(value)
    dim = ndims(value)+1;
    mergeable = true;
    if options.ColMajor
      if all(cellfun(@isscalar, value))
        dim = 1;
        if all(cellfun(@iscell, value)) % Singleton row vector [[a],[b]].
          value = cat(2, value{:});
          mergeable = isMergeable(value);
          dim = 2;
        end
      elseif all(cellfun(@iscolumn, value))
        dim = 2;
      end
    else
      if all(cellfun(@isscalar, value))
        dim = 2;
        if all(cellfun(@iscell, value)) % Singleton col vector [[a],[b]].
          value = cat(1, value{:});
          mergeable = isMergeable(value);
          dim = 1;
        end
      elseif all(cellfun(@isrow, value))
        dim = 1;
      end
    end
    if mergeable
      value = cat(dim, value{:});
    end
  end
end

function flag = isMergeable(value)
%ISMERGEABLE Check if the cell array is mergeable.
  flag = all(cellfun(@(x)isa(x, class(value{1})), value)) && ...
         all(cellfun(@(x)isequal(size(x), size(value{1})), value));
  end
