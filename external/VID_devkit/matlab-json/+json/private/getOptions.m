function [options, remaining_arguments] = getOptions(options, varargin)
%GETOPTIONS Get options from variable length arguments.
%
%    options = getOptions(options, 'option1', value1, 'option2', value2, ...);
%    [options, remaining_arguments] = getOptions(options, ...);
%
% GETOPTIONS parses variable length arguments. OPTIONS is a scalar struct of
% default values. Its fields are updated with name-value pairs of variable
% length input arguments.
%
% The name of the option matches regardless of the case, and an underscore is
% ignored. For example, the option name 'OptionName1' matches 'option_name_1'.
%
% When an option value is a scalar logical value, the value field may be
% omitted. In that case, the value is implicitly assummed TRUE.
%
% Example
% -------
%
% Getting an option struct. Invalid options will raise an error.
%
%    function myFunction(arg1, arg2, varargin)
%    %MYFUNCTION Example usage in a function body.
%    options = struct(...
%        'option1', true, ...
%        'option2', 'linear', ...
%        'option3', true, ...
%        );
%    options = getOptions(options, varargin{:});
%
%    if options.option1
%        ...
%    end
%
%    % Example usage of the function
%    myFunction(arg1, arg2, 'option1', false, 'option2', 'cubic');
%
% Chaining getOptions. Invalid options will be ignored.
%
%    [options, varargin] = getOptions(options, varargin{:});
%    [another_options, varargin] = getOptions(another_options, varargin{:});
%    somefunction(input_value, varargin{:});
%
% Skipping logical scalar values. Following examples are equivalent.
%
%    myFunction(arg1, arg2, 'option1', 'option3')
%    myFunction(arg1, arg2, 'option1', true, 'option3')
%    myFunction(arg1, arg2, 'option1', 'option3', true)
%    myFunction(arg1, arg2, 'option1', true, 'option3', true)
%
% See also varargin
  error(nargchk(1, inf, nargin, 'struct'));
  assert(isstruct(options) || isobject(options), ...
         'Expected struct or object: %s.', class(options));
  assert(isscalar(options), 'Options must be a scalar struct.');
  fields = getFields(options);
  normalized_fields = normalizeString(fields);
  options_to_delete = false(size(varargin));
  i = 1;
  while i <= numel(varargin)
    option_name = varargin{i};
    i = i + 1;
    assert(ischar(option_name), ...
           'Option name must be a string: %s.', ...
           class(option_name));
    index = find(strcmp(normalizeString(option_name), normalized_fields), 1);
    if ~isempty(index)
      default_value = options.(fields{index});
      % If it is a logical scalar, we may omit its value.
      if islogical(default_value) && ...
         isscalar(default_value) && ...
         (i > numel(varargin) || ischar(varargin{i}))
        options.(fields{index}) = true;
        options_to_delete(i-1) = true;
      else
        assert(i <= numel(varargin), 'Missing value for ''%s''.', option_name);
        if isnumeric(default_value) || islogical(default_value)
          options.(fields{index}) = feval(class(default_value), varargin{i});
        else
          assert(isa(varargin{i}, class(default_value)), ...
                 'Incompatible input type: %s for %s.', ...
                 class(varargin{i}), ...
                 class(default_value));
          options.(fields{index}) = varargin{i};
        end
        options_to_delete(i-1:i) = true;
        i = i + 1;
      end
    elseif nargout < 2
      error('Invalid option ''%s''.', option_name);
    else
      i = i + 1;
    end
  end
  remaining_arguments = varargin(~options_to_delete);
end

function fields = getFields(options)
%GETFIELDS Get field names for struct or object array.
  if isstruct(options)
    fields = fieldnames(options);
  else
    fields = properties(options);
  end
end

function normalized_name = normalizeString(name)
%NORMALIZESTRING Remove any underscore. 'option_1' matches 'Option1'.
  normalized_name = strrep(lower(name), '_', '');
end