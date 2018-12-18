function rel_path = relativepath(tgt_path, act_path)
%RELATIVEPATH  returns the relative path from an actual path to the target path.
%   Both arguments must be strings with absolute paths.
%   The actual path is optional, if omitted the current dir is used instead.
%   In case the volume drive letters don't match, an absolute path will be returned.
%   If a relative path is returned, it always starts with './' or '../'
%
%   Syntax:
%      rel_path = RELATIVEPATH(target_path, actual_path)
%   
%   Parameters:
%      target_path        - Path which is targetted
%      actual_path        - Start for relative path (optional, default = current dir)
%
%   Examples:
%      relativepath('/local/data/matlab', '/local') = './data/matlab/'
%      relativepath('/MyProject/', '/local')        = '/myproject\'
%
%      relativepath('/local/data/matlab', pwd) is the same as
%      relativepath('/local/data/matlab')
%
%   See also:  ABSOLUTEPATH PATH

%   Jochen Lenz
%   Modified by Ken Chatfield to make case sensitive
%            and to also support filenames for the tgt_path parameter


% 2nd parameter is optional:
if nargin < 2
    act_path = pwd;
end

% Predefine return string:
rel_path = '';

% Ensure act_path ends with a filesep character
if isempty(act_path) || ~isequal(act_path(end), filesep)
    act_path = [act_path filesep];
end

% If there is a file with an extension, save it for later
[tgt_path, tgt_fname, tgt_ext] = fileparts(tgt_path);
if isempty(tgt_ext)
    % treat extensionless files as part of the path
    tgt_path = fullfile(tgt_path, tgt_fname);
    tgt_fname = '';
else
    tgt_fname = [tgt_fname tgt_ext];
end
% Ensure tgt_path ends with a filesep character
if isempty(tgt_path) || ~isequal(tgt_path(end), filesep)
    tgt_path = [tgt_path filesep];
end

% Create a cell-array containing the directory levels
act_path_cell = pathparts(act_path);
tgt_path_cell = pathparts(tgt_path);

% If volumes are different, return absolute path on Windows
process_paths = true;
if ispc
   if ~isequal(act_path_cell{1}, tgt_path_cell{1})
       rel_path = tgt_path;
       process_paths = false;
   end
end

if process_paths
    % Remove level by level, as long as both are equal
    while ~isempty(act_path_cell)  && ~isempty(tgt_path_cell)
        if isequal(act_path_cell{1}, tgt_path_cell{1})
            act_path_cell(1) = [];
            tgt_path_cell(1) = [];
        else
            break
        end
    end

    % As much levels down ('../') as levels are remaining in "act_path"
    rel_path = [repmat(['..' filesep],1,length(act_path_cell)), rel_path];

    % Relative directory levels to target directory:
    rel_dirs_cell = cellfun(@(x) [x filesep], tgt_path_cell, 'UniformOutput', false);
    rel_dirs = [rel_dirs_cell{:}];
    rel_path = [rel_path rel_dirs];

    % Start with '.' or '..' :
    if isempty(rel_path)
        rel_path = ['.' filesep];
    elseif ~isequal(rel_path(1),'.')
        rel_path = ['.' filesep rel_path];
    end
end

% add on the original filename if appropriate
rel_path = fullfile(rel_path, tgt_fname);

function path_cell = pathparts(path_str)
    path_str = [filesep path_str filesep];
    path_cell = regexp(path_str, filesep, 'split');
    path_cell(strcmp(path_cell, '')) = [];
end

end