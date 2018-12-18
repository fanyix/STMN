function startup(varargin)
%STARTUP Initialize runtime environment.
%
%   json.startup(optionName, optionValue, ...)
%
% The function internally adds dynamic java class path. This process clears
% any matlab internal states, such as global/persistent variables or mex
% functions. To avoid unexpected state reset, execute this function once before
% using other json API functions.
%
% OPTIONS
%
% The function takes a following option.
%
%   'WarnOnAddPath'   Warn when javaaddpath is internally called. Default false.
%
% EXAMPLE
%
%   >> json.startup
%
% See also javaaddpath javaclasspath startup

  root = fileparts(fileparts(mfilename('fullpath')));
  jarfile = fullfile(root, 'java', 'json.jar');

  if ~any(strcmp(jarfile, javaclasspath))
    options.WarnOnAddPath = false;
    options = getOptions(options, varargin{:});

    % Check runtime.
    error(javachk('jvm'));
    assert(exist(jarfile, 'file') > 0, 'File not found: %s', jarfile);

    % Add the JAR file to the java path.
    javaaddpath(jarfile);

    if options.WarnOnAddPath
      warning('json:startup', ['Adding json.jar to the dynamic Java class ' ...
        'path. This has cleared matlab internal states, such as global '...
        'variables, persistent variables, or mex functions. To avoid this, '...
        'call json.startup before using other json API functions. See '...
        '<a href="matlab:doc javaaddpath">javaaddpath</a> for more ' ...
        'information.' ...
        ]);
    end
  end
end
