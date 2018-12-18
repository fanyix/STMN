
load('/home/SSD2/fanyi-data/ILSVRC2015/Annotations/VID/test/full.mat');
batch_size = 100;
finished = 0;
batch_count = 0;

while finished < length(annotation)
    end_idx = min(finished+batch_size, length(annotation));
    batch_anno = annotation(finished+1:end_idx);
    filename = sprintf('/home/SSD2/fanyi-data/ILSVRC2015/Annotations/VID/test/json/%g.json', batch_count);
    json.write(batch_anno, filename);
    batch_count = batch_count + 1;
    finished = end_idx;
end
