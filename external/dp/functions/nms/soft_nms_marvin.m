function bbs = soft_nms_marvin(boxes,overlap,sigma,method)
%%boxes为一个m*n的矩阵，其中m为boundingbox的个数，n的前4列为每个boundingbox的坐标，格式为
%%（x1,y1,x2,y2）；第5列为置信度。overlap为设定值，0.3,0.5 .....
%method值为1：线性，2：高斯加权，3：传统NMS
if (nargin<3)
    sigma=0.5;
    %Nt=0.8;
    method=2;
end
N=size(boxes,1);

x1 = boxes(:,1);%所有boundingbox的x1坐标
y1 = boxes(:,2);%所有boundingbox的y1坐标
x2 = boxes(:,3);%所有boundingbox的x2坐标
y2 = boxes(:,4);%所有boundingbox的y2坐标
area = (x2-x1+1) .* (y2-y1+1); %每个%所有boundingbox的面积

for ib=1:N
    tBD=boxes(ib,:);
    tscore=tBD(5);
    pos=ib+1;
    
    [maxscore,maxpos]=max(boxes(pos:end,5));
    if tscore<maxscore
        %maxscore=score;
        boxes(ib,:)=boxes(maxpos+ib,:);
        boxes(maxpos+ib,:)=tBD;
        tBD=boxes(ib,:);
        tempAera=area(ib);
        area(ib)=area(maxpos+ib);
        area(maxpos+ib)=tempAera;
    end

    xx1=max(tBD(1),boxes(pos:end,1));
    yy1=max(tBD(2),boxes(pos:end,2));
    xx2=min(tBD(3),boxes(pos:end,3));
    yy2=min(tBD(4),boxes(pos:end,4));
    
    tarea=area(ib);
    w = max(0.0, xx2-xx1+1);
    h = max(0.0, yy2-yy1+1);
    
    inter = w.*h;
    o = inter ./ (tarea + area(pos:end) - inter);%计算得分最高的那个boundingbox和其余的boundingbox的交集面积
    
    if method==1    %linear
        weight=ones(size(o));
        weight(o>overlap)=1-o;
    end
    if method==2    %guassian
        weight=exp((-o.*o)./sigma);
    end
    if method==3   %original NMS
        weight=ones(size(o));
        weight(o>overlap)=0;
    end
    boxes(pos:end,5)=boxes(pos:end,5).*weight;
end
%bbs=boxes(boxes(:,5)>threshold,:);
maximum=min(15000,size(boxes,1));
bbs=boxes(1:maximum,:);
end

