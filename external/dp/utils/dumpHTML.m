function dumpHTML(desthtml,imgList,imgStr,code,prevHtml,nextHtml,N,hgt,wid)
if length(imgList) ~= length(imgStr)
    error('mismatch between image and string');
end
if numel(N)==1
    N=N*ones(ceil(length(imgList)/N),1);
end
if isempty(code)
    % there is no extra code for the images
    code=cell(length(imgList),1);
elseif ~iscell(code)
    code=repmat({code},[length(imgList) 1]);
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
imgCounter=1;
while i<=length(imgList)
    NLeft = min(length(imgList) - i + 1,N(counter));
    % print texts
    html{end+1}=sprintf('<tr>\n');
    for j = 0:(NLeft-1)
        html{end+1}=sprintf('<td>');
        html{end+1}=imgStr{i+j};
        html{end+1}=sprintf('</td>');
        html{end+1}=sprintf('    ');
    end
    html{end+1}=sprintf('\n</tr>\n');
    
    % print images
    html{end+1}=sprintf('<tr>\n');
    for j = 0:(NLeft-1)
        html{end+1}=sprintf('<td><canvas id="myCanvas_%g',imgCounter);
        %html{end+1}=imgList{i+j};
        html{end+1}=sprintf('" height="%g" width="%g"/></td>',hgt,wid);
        html{end+1}=sprintf('    ');
        imgCounter=imgCounter+1;
    end
    html{end+1}=sprintf('\n</tr>\n');
    html{end+1}=sprintf('<p></p>');
    i=i+NLeft;
    counter=counter+1;
end
html{end+1}=sprintf('</table>\n');
html{end+1}=sprintf('<a href="%s">previous</a> &nbsp;&nbsp <a href="%s">next</a>\n',prevHtml,nextHtml);

% output "script" section
imgCounter=imgCounter-1;
html{end+1}=sprintf('<script>\n');
html{end+1}=sprintf('// ----- script section begins\n');
for i = 1:imgCounter
    html{end+1}=sprintf('var c=document.getElementById("myCanvas_%g");\n',i);
    html{end+1}=sprintf('var ctx_%g=c.getContext("2d");\n',i);
    html{end+1}=sprintf('var imageObj_%g = new Image();\n',i);
    html{end+1}=sprintf('imageObj_%g.src=''%s'';\n',i,imgList{i});
    html{end+1}=sprintf('imageObj_%g.onload = function() {\n',i);
    html{end+1}=sprintf('ctx_%g.drawImage(imageObj_%g, 0, 0,%g,%g);\n',i,i,wid,hgt);
    if ~isempty(code{i})
        tmpCode = strrep(code{i},'ctx', sprintf('ctx_%g',i));
    else
        tmpCode=[];
    end
    html{end+1}=sprintf('%s\n',tmpCode);
    html{end+1}=sprintf('ctx_%g.stroke();\n',i);
    html{end+1}=sprintf('};\n');
    html{end+1}=sprintf('\n');
end
html{end+1}=sprintf('// ----- script section ends\n');
html{end+1}=sprintf('</script>\n');
html{end+1}=sprintf('</body></html>\n');
html = cell2mat(html);
h = fopen(desthtml,'a+');
fprintf(h,html);
fclose(h);