function im = plot_bb(im,bbox,color)
if ndims(im)==2
    im=repmat(im,[1 1 3]);
end

im_bb = im*0+1;
width = 3;
h = size(im,1);
w = size(im,2);

if ischar(color)
    switch color
        case 'r'
            color = 1;
        case 'g'
            color = 2;
        case 'b'
            color = 3;
        otherwise
            error('unknown color!\n');
    end
end
        
im_bb = im;
bbox = clip_to_image(round(bbox(1:4)),[1 1 size(im,2) size(im,1)]);

if ndims(im_bb) == 2
    im_bb(threshY(im,bbox(2)-width):threshY(im,bbox(2)+width),threshX(im,bbox(1)):threshX(im,bbox(3))) = 0;
    im_bb(threshY(im,bbox(4)-width):threshY(im,bbox(4)+width),threshX(im,bbox(1)):threshX(im,bbox(3))) = 0;
    im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(1)-width):threshX(im,bbox(1)+width)) = 0;
    im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(3)-width):threshX(im,bbox(3)+width)) = 0;
else
    im_bb(threshY(im,bbox(2)-width):threshY(im,bbox(2)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),:) = 0;
    im_bb(threshY(im,bbox(4)-width):threshY(im,bbox(4)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),:) = 0;
    im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(1)-width):threshX(im,bbox(1)+width),:) = 0;
    im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(3)-width):threshX(im,bbox(3)+width),:) = 0;
    if numel(color) == 1
        im_bb(threshY(im,bbox(2)-width):threshY(im,bbox(2)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),color) = 255;
        im_bb(threshY(im,bbox(4)-width):threshY(im,bbox(4)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),color) = 255;
        im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(1)-width):threshX(im,bbox(1)+width),color) = 255;
        im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(3)-width):threshX(im,bbox(3)+width),color) = 255;
    else
        for i = 1:length(color)
            im_bb(threshY(im,bbox(2)-width):threshY(im,bbox(2)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),i) = color(i)*255;
            im_bb(threshY(im,bbox(4)-width):threshY(im,bbox(4)+width),threshX(im,bbox(1)):threshX(im,bbox(3)),i) = color(i)*255;
            im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(1)-width):threshX(im,bbox(1)+width),i) = color(i)*255;
            im_bb(threshY(im,bbox(2)):threshY(im,bbox(4)),threshX(im,bbox(3)-width):threshX(im,bbox(3)+width),i) = color(i)*255;
        end
    end
end
im = im_bb;
% im = im.*im_bb;

function x=threshX(im,x)
x=min(max(x,1),size(im,2));

function y=threshY(im,y)
y=min(max(y,1),size(im,1));





