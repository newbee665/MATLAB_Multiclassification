function [RANKED, WEIGHT] = IG(X, Y , TT, verbose )

if (nargin < 3)
    verbose = 0;
    TT = 3;
end
if (nargin < 4)
    verbose = 0;
end

A = LearningGraphWeights(X, Y, TT, verbose );

priori_len = ceil(max( A*ones(length(A),1)))/size(X,2);

factor = 0.99;
 
if (verbose)
    fprintf('4) Letting paths tend to infinite \n');
end
I = eye( size( A ,1 )); % Identity Matrix
rho = max(eig(A));
r = factor/rho; % Set a meaningful value for r
y = I - ( r * A );

S = inv( y ) - I; % see Gelfand's formula - convergence of the geometric series of matrices


if (verbose)
    fprintf('5) Estimating relevancy scores \n');
end
WEIGHT = sum( S , 2 ); % prob. scores s(i)


if (verbose)
    fprintf('6) Features ranking  \n');
end
[~ , RANKED ]= sort( WEIGHT , 'descend' );

RANKED = RANKED';
WEIGHT = WEIGHT';


end

function [G] = LearningGraphWeights( train_x , train_y, TT, verbose )

if (verbose)
    fprintf(['\n+ PAMI - Feature selection: inf-VR \n' ...
        ' This procedure (PLSA-like) aims to discover something abount the meaning \n' ...
        ' behind the tokens, about more complex latent constructs in the\n' ...
        ' features distribution. Latent variables - factors - are combinations of observed\n' ...
        ' values (tokens) which can co-occur in different features.\n' ...
        ' Different from PLSA, the presence of a value in two or more feature\n' ...
        ' distributions cannot be subject to:\n' ...
        ' Polysemy: the same (observed) token have a unique meaning\n' ...
        '           (e.g., f(1) 1 -> good sample, f(2) 1 -> the same)\n' ...
        ' Or, \n' ...
        ' Synonymy: different tokens cannot have the same meaning:\n'...
        '           (e.g., f(1) 1 -> good sample, f(2) 2 -> bad sample)\n']);
end
if (nargin<3)
    verbose = 0;
end

params.iter = 15;

numFeat = size(train_x,2);
numSamp = size(train_x,1);

num_classes = length(unique(train_y));
unique_class_labels = sort(unique(train_y),'descend');
right_labels = 999*ones(size(train_y));

for ii=1:num_classes
    right_labels(train_y == unique_class_labels(ii)) = ii;
end

for ii=1:num_classes
    
    c{ii,1} = train_x(right_labels==ii,:);
    mu_c(ii,:) = mean(c{ii});
    st(ii,:)   = std(c{ii}).^2;
    
end

st(st<1000000*eps)=1;
              

numFactors = 2; % Num of Latent Variables - F1: Relevant Feature, F2: Not Relevant

% We define a "dictionary" or "vocabulary" as a row vector of 'numTokens'.
% Note, this first operation is a mapping from the raw features values to a
% first level analysis where the allowed tokes are:
% N: data is well represented (if the sample is closer to its correct class mean)
% 1: bad sample (otherwise)
token = [1:TT]; % the order of the tokens (multinominal vars) in the dictionary is fixed.
flag = 1;
numTokens = length(token);
% this dictionary can also be augmented with more symbols for intermediate positions.

% Given the dictionary, we consider each feature vector as a document,
% A feature (document) is a vector where each entry corresponds to a
% different token (term) and the number at that entry corresponds to how many times
% that token (term) was present in the feature distribution.
tokenFeatMatrix = zeros(numTokens,numFeat);
priors_sep_scores = zeros(1,numFeat);
% In order to create the token by feature matrix, we consider for each
% sample in the feature distribution its closest class mean, in order to
% map each value to a very simple representation.
for f=1:numFeat
    d = ( bsxfun( @minus, train_x(:,f), mu_c(:,f)' )).^2 / sum( st(:,f) );  % Multi-class fisher scores
    prob_class_est = (abs(d)./repmat(sum(abs(d),2),[1,size(d,2)]));
    prob_class = zeros(numSamp,1);
    for ss=1:numSamp
        prob_class(ss,1) = prob_class_est(ss,right_labels(ss));
    end
    if ~isnan(sum(prob_class)) && ~isinf(sum(prob_class))
        tokenFeatMatrix(:,f) = histc(prob_class,linspace(min(prob_class),max(prob_class),TT));
    end
    priors_sep_scores(f) = [sum(tokenFeatMatrix(TT:-1:(TT-round(TT*0.35)),f))/numSamp];  

end
priors_sep_scores = 100*priors_sep_scores;

% Check and remove unwanted values
priors_sep_scores(isnan(priors_sep_scores)) = 1;
priors_sep_scores(isinf(priors_sep_scores)) = 1;       

prob_token_factor = [linspace(5000,1,numTokens)', linspace(1,5000,numTokens)'] ; %  p(token | factor): Factor 1 will represent the discriminative topic
for z = 1:numFactors
    prob_token_factor(:, z) = prob_token_factor(:, z) / sum(prob_token_factor(:, z)); % normalization
end

% Initialize  p(factor | feat)
% Since Factor 1 represent the discriminative topic, we set high scores of good features on
% F1 and high scores for bad features on F2
prob_factor_feat = zeros(numFactors,numFeat); % init
% F1 Relevant: high scores for discriminative features
prob_factor_feat(1,:) = priors_sep_scores ; 
% F1 Irrelevant: high scores for unwanted features
prob_factor_feat(2,:) = 100-prob_factor_feat(1,:); 
 
for z = 1:numFeat
    prob_factor_feat(:, z) = prob_factor_feat(:, z) / sum(prob_factor_feat(:, z)); % normalization
end

% Initialize also  p(token | feat)
prob_token_feat = zeros(numTokens, numFeat); % p(token | feat)
for f = 1:numFeat
    for z = 1:numFactors
        prob_token_feat(:, f) = prob_token_feat(:, f) + ...
            prob_token_factor(:, z) .* prob_factor_feat(z, f);
    end
    assert(sum(prob_token_feat(:, f)) - 1.0 < 1e-6);
end

prob_factor_token_feat = cell(numFactors, 1);   % p(factor | feat, token)
for z = 1 : numFactors
    prob_factor_token_feat{z} = zeros(numTokens, numFeat);
end


% The implementation of PLSA + EM algorithm is based on the code at:
% https://github.com/lizhangzhan/plsa
% https://github.com/lizhangzhan/plsa/blob/master/plsa.m

%% 2) Expectation-Maximization: maximum log-likelihood estimations
if (verbose)
    disp('Expectation-Maximization: maximum log-likelihood estimations.');
end
lls = -999; % maximum log-likelihood estimations
for ii = 1 : params.iter
    
    if (verbose)
        fprintf('+ E-step:\nP(factor | token,feat) =  [ P(factor | feat) * P(token | factor) ] / P(token | feat )   \n');
    end
    for f = 1:numFeat
        %fprintf('processing doc %d\n', f);
        t = find(tokenFeatMatrix(:, f));
        for z = 1:numFactors
            prob_factor_token_feat{z}(t, f) = prob_factor_feat(z, f) .* prob_token_factor(t, z) ./ prob_token_feat(t, f);
        end
    end
    
    if (verbose)
        fprintf('+ M-step: Update P(factor* | feat) = 1/Z * SUM_token ( tokenFeatMatrix(token, feat) * P(factor* | token, feat) ) \n');
    end
    for f = 1:numFeat
        t = find(tokenFeatMatrix(:, f)); % which tokens occur in feature f
        for z = 1:numFactors
            prob_factor_feat(z, f) = sum(  tokenFeatMatrix(t, f) .* prob_factor_token_feat{z}(t, f) );
        end
        NC = sum(prob_factor_feat(:, f));
        if NC==0
            NC=0.00001;
        end
        prob_factor_feat(:, f) = prob_factor_feat(:, f) / NC;
    end
    % if too many tokens you may divide by zero -> NaN
    %     prob_factor_feat(isnan(prob_factor_feat)) = 0;
    if (verbose)
        disp('+ Update p(tokens | factor) ');
    end
    for z = 1:numFactors
        for t = 1:numTokens
            f = find(tokenFeatMatrix(t, :)); % in which features token t occurs
            prob_tokens_factor(t, z) = sum(tokenFeatMatrix(t, f) .* prob_factor_token_feat{z}(t, f));
        end
        prob_tokens_factor(:, z) = prob_tokens_factor(:,z) / sum(prob_tokens_factor(:,z));
    end
    
    % M-Step - Update prob_token_factor
    mat_prob_token_factor_feat = permute(cell2mat(reshape(prob_factor_token_feat,1,1,[])),[3 1 2]);
    prob_token_factor = squeeze(sum(mat_prob_token_factor_feat,3))';
    for z = 1:numFactors
        prob_token_factor(:, z) = prob_token_factor(:, z) / sum(prob_token_factor(:, z)); % normalization
    end

    % calculate likelihood and update p(term, doc)
    if (verbose)
        fprintf('+ Iteration %d\n', ii);
        disp(' + Calculate maximum likelihood...');
    end
    ll = 0;
    for f = 1:numFeat
        prob_token_feat(:, f) = 0;
        for z = 1:numFactors
            prob_token_feat(:, f) = prob_token_feat(:, f) + ...
                prob_factor_feat(z, f) .* prob_token_factor(:, z);
        end
        assert(sum(prob_token_feat(:, f)) - 1.0 < 1e-6);
        t = find(tokenFeatMatrix(:, f));
        ll = ll + sum(tokenFeatMatrix(t, f) .* log(prob_token_feat(t, f)));
    end
    
    if (verbose)
        fprintf('likelihood: %f\n', ll);
    end
    
    lls = [lls;ll];
    
    if (verbose)
        plot(lls,'r','linewidth',2);
        grid on
        title 'EM - Maximum likelihood';
        drawnow
    end
    
    if abs(lls(end)-lls(end-1)) < 0.000001 
        break;
    end
    
end
% EM Completed.
if (verbose)
    fprintf('Optimized: %.2f \n',abs(lls(end)-lls(1)));
end
factor_representing_relevancy = 1;

% Building the graph:
% In order to connect features, we compute the joint probability
% P( factor_1 | feat_i, feat_j ) = P( factor_1 | feat_i ) * P( factor_1 | feat_j )
G = prob_factor_feat(factor_representing_relevancy,:)'*prob_factor_feat(factor_representing_relevancy,:);

end

