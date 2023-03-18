function [fea, score] = chisquare(X_train, Y_train, K)

% bdisp=0;

nd = size(X_train,2);
nc = size(X_train,1);

% t1=cputime;
for i=1:nd
   t(i) = mutualinfo(X_train(:,i), Y_train);
end

[tmp, idxs] = sort(-t);
fea_base = idxs(1:K);

fea(1) = idxs(1);

KMAX = min(1000,nd); 

idxleft = idxs(2:KMAX);

k=1;


for k=2:K
%    t1=cputime;
   ncand = length(idxleft);
   curlastfea = length(fea);
   for i=1:ncand
      t_mi(i) = mutualinfo(X_train(:,idxleft(i)), Y_train); %又算了一遍特征与标签的互信息
      mi_array(idxleft(i),curlastfea) = getmultimi(X_train(:,fea(curlastfea)), X_train(:,idxleft(i)));%特征间的互信息矩阵
      c_mi(i) = mean(mi_array(idxleft(i), :)); %对特征间的互信息取平均值
   end

   [score(k), fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));%特征与标签的互信息减去特征间的互信息，[最大值,定位]

   tmpidx = fea(k); 
   fea(k) = idxleft(tmpidx); 
   idxleft(tmpidx) = [];
   
%    if bdisp==1,
% %    fprintf('k=%X_train cost_time=%5.4f cur_fea=%X_train #left_cand=%X_train\n', ...
%       k, cputime-t1, fea(k), length(idxleft));
%    end;
end

return;

function c = getmultimi(da, dt) 
for i=1:size(da,2)
   c(i) = mutualinfo(da(:,i), dt);
end