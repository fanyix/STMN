function nms_boxes = soft_nms(boxes, sigma, thresh)

if ~exist('sigma', 'var')
    sigma = 0.5;
end

if ~exist('thresh', 'var')
    thresh = 1e-6;
end

nms_boxes = boxes;
if isempty(boxes)
  return;
end

N = size(nms_boxes, 1);
ov = boxoverlap(nms_boxes, nms_boxes);
index = [1: N];  % index(i)=j --> the original j'th row of boxes in currently at i'th row
cursor = 1;
while cursor < N
    % get the max
    [ignore, maxpos] = max(nms_boxes(cursor:N, 5));
    maxpos = cursor + maxpos - 1;
    % swap
    tmp_row = nms_boxes(cursor, :);
    nms_boxes(cursor, :) = nms_boxes(maxpos, :);
    nms_boxes(maxpos, :) = tmp_row;
    tmp_idx = index(cursor);
    index(cursor) = index(maxpos);
    index(maxpos) = tmp_idx;
    % get cur_ov
    cur_ov = ov(index(cursor), index((cursor+1):N));
    weights = exp(-(cur_ov.*cur_ov)/sigma);
    nms_boxes((cursor+1):N, 5) = nms_boxes((cursor+1):N, 5) .* reshape(weights, [numel(weights), 1]);
    % prune
    prune_idx = nms_boxes((cursor+1):N, 5) < thresh;
    if any(prune_idx)
        prune_idx = find(prune_idx);
        prune_idx = prune_idx + cursor;
        nms_boxes(prune_idx, :) = [];
        index(prune_idx) = [];
        N = N - numel(prune_idx);
    end
    % increment cursor
    cursor = cursor + 1;
end





