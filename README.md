# spdnetcopy
```MATLAB
function [net, info] = spdnet_afew(varargin)
%set up the path
confPath;
%parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'spddb_afew_train_spd400_int_histeq.mat');
opts.batchSize = 30 ;
opts.test.batchSize = 1;
opts.numEpochs = 500 ;
opts.gpus = [] ;
opts.learningRate = 0.01*ones(1,opts.numEpochs);
opts.weightDecay = 0.0005 ;
opts.continue = 1;
%spdnet initialization
net = spdnet_init_afew() ;
%loading metadata 
load(opts.imdbPathtrain) ;
%spdnet training
[net, info] = spdnet_train_afew(net, spd_train, opts);
```
```MATLAB
function net = spdnet_init_afew(varargin)
% spdnet_init initializes a spdnet

rng('default');
rng(0) ;

opts.layernum = 3;

Winit = cell(opts.layernum,1);
opts.datadim = [400, 200, 100, 50];


for iw = 1 : opts.layernum
    A = rand(opts.datadim(iw));
    [U1, S1, V1] = svd(A * A');
    Winit{iw} = U1(:,1:opts.datadim(iw+1));
end

f=1/100 ;
classNum = 7;
fdim = size(Winit{iw},2)*size(Winit{iw},2);
theta = f*randn(fdim, classNum, 'single');
Winit{end+1} = theta;

net.layers = {} ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{1}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{2}) ;
net.layers{end+1} = struct('type', 'rec') ;
net.layers{end+1} = struct('type', 'bfc',...
                          'weight', Winit{3}) ;
net.layers{end+1} = struct('type', 'log') ;
net.layers{end+1} = struct('type', 'fc', ...
                           'weight', Winit{end}) ;
net.layers{end+1} = struct('type', 'softmaxloss') ;
```
```MATLAB

function [net, info] = spdnet_train_afew(net, spd_train, opts)

opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ;
opts.val = find(spd_train.spd.set==2) ;
    
for epoch=1:opts.numEpochs
    learningRate = opts.learningRate(epoch);
    
     % fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
     if opts.continue
         if exist(modelPath(epoch),'file')
             if epoch == opts.numEpochs
                 load(modelPath(epoch), 'net', 'info') ;
             end
             continue ;
         end
         if epoch > 1
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
         end
     end
    
    train = opts.train(randperm(length(opts.train))) ; % shuffle
    val = opts.val;

    [net,stats.train] = process_epoch(opts, epoch, spd_train, train, learningRate, net) ;
    [net,stats.val] = process_epoch(opts, epoch, spd_train, val, 0, net) ;

   
    % save
    evaluateMode = 0;
    
    if evaluateMode, sets = {'train'} ; else sets = {'train', 'val'} ; end

    for f = sets
        f = char(f) ;
        n = numel(eval(f)) ; %
        info.(f).objective(epoch) = stats.(f)(2) / n ;
        info.(f).error(:,epoch) = stats.(f)(3:end) / n ;
    end
    if ~evaluateMode, save(modelPath(epoch), 'net', 'info') ; end

    figure(1) ; clf ;
    hasError = 1 ;
    subplot(1,1+hasError,1) ;
    if ~evaluateMode
        semilogy(1:epoch, info.train.objective, '.-', 'linewidth', 2) ;
        hold on ;
    end
    semilogy(1:epoch, info.val.objective, '.--') ;
    xlabel('training epoch') ; ylabel('energy') ;
    grid on ;
    h=legend(sets) ;
    set(h,'color','none');
    title('objective') ;
    if hasError
        subplot(1,2,2) ; leg = {} ;
        if ~evaluateMode
            plot(1:epoch, info.train.error', '.-', 'linewidth', 2) ;
            hold on ;
            leg = horzcat(leg, strcat('train ', opts.errorLabels)) ;
        end
        plot(1:epoch, info.val.error', '.--') ;
        leg = horzcat(leg, strcat('val ', opts.errorLabels)) ;
        set(legend(leg{:}),'color','none') ;
        grid on ;
        xlabel('training epoch') ; ylabel('error') ;
        title('error') ;
    end
    drawnow ;
    print(1, modelFigPath, '-dpdf') ;
end
    
    
    
function [net,stats] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net)

training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end

stats = [0 ; 0 ; 0] ;
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;


for ib = 1 : batchSize : length(trainInd)
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ;
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;
    else
        batchSize_r = batchSize;
    end
    spd_data = cell(batchSize_r,1);    
    spd_label = zeros(batchSize_r,1);
    for ib_r = 1 : batchSize_r
        spdPath = [spd_train.spdDir '\' spd_train.spd.name{trainInd(ib+ib_r-1)}];
        load(spdPath); spd_data{ib_r} = Y1;
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));

    end
    net.layers{end}.class = spd_label ;

    %forward/backward spdnet
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_myforbackward(net, spd_data, dzdy, res) ;
   
    %accumulating graidents
    if numGpus <= 1
      [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res) ;
    else
      if isempty(mmap)
        mmap = map_gradients(opts.memoryMapFile, net, res, numGpus) ;
      end
      write_gradients(mmap, net, res) ;
      labBarrier() ;
      [net,res] = accumulate_gradients(opts, learningRate, batchSize_r, net, res, mmap) ;
    end
          
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;

    numDone = numDone + batchSize_r ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    stats = stats+[batchTime ; res(end).x ; error]; % works even when stats=[]
    
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;

    fprintf(' error: %.5f', stats(3)/numDone) ;
    fprintf(' obj: %.5f', stats(2)/numDone) ;

    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf('\n') ;
    
end


% -------------------------------------------------------------------------
function [net,res] = accumulate_gradients(opts, lr, batchSize, net, res, mmap)
% -------------------------------------------------------------------------
for l=numel(net.layers):-1:1
  if isempty(res(l).dzdw)==0
    if ~isfield(net.layers{l}, 'learningRate')
       net.layers{l}.learningRate = 1 ;
    end
    if ~isfield(net.layers{l}, 'weightDecay')
       net.layers{l}.weightDecay = 1;
    end
    thisLR = lr * net.layers{l}.learningRate ;

    if isfield(net.layers{l}, 'weight')
        if strcmp(net.layers{l}.type,'bfc')==1
            W1=net.layers{l}.weight;
            W1grad = (1/batchSize)*res(l).dzdw;
            %gradient update on Stiefel manifolds
            problemW1.M = stiefelfactory(size(W1,1), size(W1,2));
            W1Rgrad = (problemW1.M.egrad2rgrad(W1, W1grad));
            net.layers{l}.weight = (problemW1.M.retr(W1, -thisLR*W1Rgrad)); %%!!!NOTE
        else
            net.layers{l}.weight = net.layers{l}.weight - thisLR * (1/batchSize)* res(l).dzdw ;
        end
    
    end
  end
end
```
```MATLAB
% Add folders to path.
addpath(pwd);

cd manopt;
addpath(genpath(pwd));
cd ..;

cd utils;
addpath(genpath(pwd));
cd ..;

cd spdnet;
addpath(genpath(pwd));
cd ..;
```
```MATLAB
function [Y, Y_w] = vl_mybfc(X, W, dzdy)
%[DZDX, DZDF, DZDB] = VL_MYBFC(X, F, B, DZDY)
%BiMap layer

Y = cell(length(X),1);
for ix = 1  : length(X)
    Y{ix} = W'*X{ix}*W;
end
if nargin == 3
    [dim_ori, dim_tar] = size(W);
    Y_w = zeros(dim_ori,dim_tar);
    for ix = 1  : length(X)
        if iscell(dzdy)==1
            d_t = dzdy{ix};
        else
            d_t = dzdy(:,ix);
            d_t = reshape(d_t,[dim_tar dim_tar]);
        end
        Y{ix} = W*d_t*W';
        Y_w = Y_w+2*X{ix}*W*d_t;
    end
end
```
```MATLAB

function [Y, Y_w] = vl_myfc(X, W, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer

X_t = zeros(size(X{1},1)^2,length(X));
for ix = 1 : length(X)
    x_t = X{ix};
    X_t(:,ix) = x_t(:);
end
if nargin < 3
    Y = W'*X_t;
else
    Y = W * dzdy;
    Y_w = X_t * dzdy';
end
```
```MATLAB
function res = vl_myforbackward(net, x, dzdy, res, varargin)
% vl_myforbackward  evaluates a simple SPDNet

opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-4;

% opts = vl_argparse(opts, varargin);

n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'bfc'
      res(i+1).x = vl_mybfc(res(i).x, l.weight) ; 
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ; 
    case 'rec'
      res(i+1).x = vl_myrec(res(i).x, opts.epsilon) ; 
    case 'log'
      res(i+1).x = vl_mylog(res(i).x) ;
    case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ; 

    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end

% -------------------------------------------------------------------------
%                                                             Backward pass
% -------------------------------------------------------------------------

if doder
  res(n+1).dzdx = dzdy ;
  for i=n:-1:max(1, n-opts.backPropDepth+1)
    l = net.layers{i} ;
    res(i).backwardTime = tic ;
    switch l.type
      case 'bfc'
        [res(i).dzdx, res(i).dzdw] = ...
             vl_mybfc(res(i).x, l.weight, res(i+1).dzdx) ;
      case 'fc'
        [res(i).dzdx, res(i).dzdw]  = ...
              vl_myfc(res(i).x, l.weight, res(i+1).dzdx) ; 
      case 'rec'
        res(i).dzdx = vl_myrec(res(i).x, opts.epsilon, res(i+1).dzdx) ;
      case 'log'
        res(i).dzdx = vl_mylog(res(i).x, res(i+1).dzdx) ;
      case 'softmaxloss'
        res(i).dzdx = vl_mysoftmaxloss(res(i).x, l.class, res(i+1).dzdx) ;
      case 'custom'
        res(i) = l.backward(l, res(i), res(i+1)) ;
    end
    if opts.conserveMemory
      res(i+1).dzdx = [] ;
    end
    if gpuMode & opts.sync
      wait(gpuDevice) ;
    end
    res(i).backwardTime = toc(res(i).backwardTime) ;
  end
end
```
```MATLAB
function Y = vl_mylog(X, dzdy)
%Y = VL_MYLOG(X, DZDY)
%LogEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);

for ix = 1 : length(X)
    [Us{ix},Ss{ix},Vs{ix}] = svd(X{ix});
end


D = size(Ss{1},2);
Y = cell(length(X),1);

if nargin < 2
    for ix = 1:length(X)
        Y{ix} = Us{ix}*diag(log(diag(Ss{ix})))*Us{ix}';
    end
else
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};
        diagS = diag(S);
        ind =diagS >(D*eps(max(diagS)));
        Dmin = (min(find(ind,1,'last'),D));
        
        S = S(:,ind); U = U(:,ind);
        dLdC = double(reshape(dzdy(:,ix),[D D])); dLdC = symmetric(dLdC);
        

        dLdV = 2*dLdC*U*diagLog(S,0);
        dLdS = diagInv(S)*(U'*dLdC*U);
        if sum(ind) == 1 % diag behaves badly when there is only 1d
            K = 1./(S(1)*ones(1,Dmin)-(S(1)*ones(1,Dmin))'); 
            K(eye(size(K,1))>0)=0;
        else
            K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))');
            K(eye(size(K,1))>0)=0;
            K(find(isinf(K)==1))=0; 
        end
        if all(diagS==1)
            dzdx = zeros(D,D);
        else
            dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U';

        end
        Y{ix} =  dzdx; %warning('no normalization');        
    end
end
```
```MATLAB

function Y = vl_myrec(X, epsilon, dzdy)
% Y = VL_MYREC (X, EPSILON, DZDY)
% ReEig layer

Us = cell(length(X),1);
Ss = cell(length(X),1);
Vs = cell(length(X),1);

for ix = 1 : length(X)
    [Us{ix},Ss{ix},Vs{ix}] = svd(X{ix});
end

D = size(Ss{1},2);
Y = cell(length(X),1);

if nargin < 3
    for ix = 1:length(X)
        [max_S, ~]=max_eig(Ss{ix},epsilon);
        Y{ix} = Us{ix}*max_S*Us{ix}';
    end
else
    for ix = 1:length(X)
        U = Us{ix}; S = Ss{ix}; V = Vs{ix};

        Dmin = D;
        
        dLdC = double(dzdy{ix}); dLdC = symmetric(dLdC);
        
        [max_S, max_I]=max_eig(Ss{ix},epsilon);
        dLdV = 2*dLdC*U*max_S;
        dLdS = (diag(not(max_I)))*U'*dLdC*U;
        
        
        K = 1./(diag(S)*ones(1,Dmin)-(diag(S)*ones(1,Dmin))');
        K(eye(size(K,1))>0)=0;
        K(find(isinf(K)==1))=0; 
        
        dzdx = U*(symmetric(K'.*(U'*dLdV))+dDiag(dLdS))*U';
        
        Y{ix} =  dzdx; %warning('no normalization');
    end
end
```
```MATLAB
function Y = vl_mysoftmaxloss(X,c,dzdy)
% Softmax layer

% class c = 0 skips a spatial location
mass = single(c > 0) ;
mass = mass';

% convert to indexes
c_ = c - 1 ;
for ic = 1  : length(c)
    c_(ic) = c(ic)+(ic-1)*size(X,1);
end

% compute softmaxloss
Xmax = max(X,[],1) ;
ex = exp(bsxfun(@minus, X, Xmax)) ;

% s = bsxfun(@minus, X, Xmax);
% ex = exp(s) ;
% y = ex./repmat(sum(ex,1),[size(X,1) 1]);

%n = sz(1)*sz(2) ;
if nargin < 3
  t = Xmax + log(sum(ex,1)) - reshape(X(c_), [1 size(X,2)]) ;
  Y = sum(sum(mass .* t,1)) ;
else
  Y = bsxfun(@rdivide, ex, sum(ex,1)) ;
  Y(c_) = Y(c_) - 1;
  Y = bsxfun(@times, Y, bsxfun(@times, mass, dzdy)) ;
end
```
