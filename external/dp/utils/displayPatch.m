function displayPatch(desthtml,patch,str,prevHtml,nextHtml,N,hgt,wid,options)
if ~exist('options','var')
    options=[];
end
if ~isfield(options,'hyperlink')
    options.hyperlink=true;
end

if length(patch) ~= length(str)
    error('mismatch between image and string');
end
if numel(N)==1
    N=N*ones(ceil(length(patch)/N),1);
end

[myDirectory] = fileparts(desthtml);
if ~exist(myDirectory,'dir')
    mkdir(myDirectory);
    fprintf('WARNING: unexisted directory, create new folder\n');
end

if isempty(prevHtml)
    prevHtml='.';
end
if isempty(nextHtml)
    nextHtml='.';
end

if ~exist('hgt','var')
    hgt = 300;
end

if ~exist('wid','var')
    wid = 300;
end

html = [];
html{end+1}=sprintf('<!DOCTYPE html>\n');
html{end+1}=sprintf('<html><body>\n');
html{end+1}=sprintf('<table>\n');
% for i = 1:N:length(patch)
i=1;
counter=1;
while i<=length(patch)
    NLeft = min(length(patch) - i + 1,N(counter));
    % print texts
    html{end+1}=sprintf('<tr>\n');
    for j = 0:(NLeft-1)
        html{end+1}=sprintf('<td>');
        html{end+1}=str{i+j};
        html{end+1}=sprintf('</td>');
        html{end+1}=sprintf('    ');
    end
    html{end+1}=sprintf('\n</tr>\n');
    
    % print images
    html{end+1}=sprintf('<tr>\n');
    for j = 0:(NLeft-1)
        html{end+1}=sprintf('<td><img src="');
        html{end+1}=patch{i+j};
        html{end+1}=sprintf('" height="%g" width="%g"/></td>',hgt,wid);
        html{end+1}=sprintf('    ');
    end
    html{end+1}=sprintf('\n</tr>\n');
    html{end+1}=sprintf('<p></p>');
    i=i+NLeft;
    counter=counter+1;
end
html{end+1}=sprintf('</table>\n');
if options.hyperlink
    html{end+1}=sprintf('<a href="%s">previous</a> &nbsp;&nbsp <a href="%s">next</a>\n',prevHtml,nextHtml);
end
html{end+1}=sprintf('</body></html>\n');
html = cell2mat(html);
h = fopen(desthtml,'a+');
fprintf(h,html);
fclose(h);