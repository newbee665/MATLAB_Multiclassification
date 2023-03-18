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
      t_mi(i) = mutualinfo(X_train(:,idxleft(i)), Y_train); %������һ���������ǩ�Ļ���Ϣ
      mi_array(idxleft(i),curlastfea) = getmultimi(X_train(:,fea(curlastfea)), X_train(:,idxleft(i)));%������Ļ���Ϣ����
      c_mi(i) = mean(mi_array(idxleft(i), :)); %��������Ļ���Ϣȡƽ��ֵ
   end

   [score(k), fea(k)] = max(t_mi(1:ncand) - c_mi(1:ncand));%�������ǩ�Ļ���Ϣ��ȥ������Ļ���Ϣ��[���ֵ,��λ]

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