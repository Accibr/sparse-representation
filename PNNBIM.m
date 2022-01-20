clc;
clear all;

% addpath drtoolbox;
addpath(genpath('.\drtoolbox'));

tic;      %start of the time
c = 38;   %number of classes of face images
ci = 20;   %number of train images each class
load('count YaleB')
load('ClassSetYaleB')
% TrainSet = [];
% TestSet = zeros(c,64-ci);
% for i=1:c
%     p = randperm(count(i));
%     TrainSet = [TrainSet;p(1:ci)];
%     TestSet(i,1:count(i)-ci) = p(ci+1:count(i));
% end
% rateC = [1/32 1/24 1/16 1/8];  %rate of down-sampling
rateC = 200;%[50 100 200 500];  %dimensiona of compress
Rate = []; %record the recognition rate
%% construct dictionary
disp(['Recognition Rate (%)']);
for Downrate=1:length(rateC)
            allsamples = []; %store all train images
            for i = 1:c
                for j=1:ci
                    faceT = imread(strcat('.\Cropped Yale\yaleB',num2str(i),'.\',num2str(TrainSet(i,j)),'.pgm'));   %read train image
                    faceV = faceT(:); %vector of train image
                    faceD = 2*double(faceV)/255-1;
                    allsamples = [allsamples faceD]; %all train images
                end
            end
    
         W = PCA(allsamples',rateC(Downrate));    %compress matrix
            [mappedX, mapping] = compute_mapping(allsamples, 'PCA', rateC(Downrate));
            W = mappedX';
            A = W*allsamples;      %dictionary after compress
%     load('Wcom200YaleB')
    [m,n] = size(A);
    for i = 1 : n
        A(:,i) = A(:,i)/norm(A(:,i)) ;  %normalize the signals to unit L2-norm
    end
    [m,n] = size(A);
    for i = 1 : n
        A(:,i) = A(:,i)/norm(A(:,i)) ;  %normalize the signals to unit L2-norm
    end
    
    invG = A'*inv(A*A');
    P = invG*A;
    I = eye(n);
    %     IP = I-P;
    
    %% read test image and recognize
    K = 5;%[37 38 39 40]; %sparsity level
    for sparse=1:length(K)
        
        t0 = tic;      %start of the time
        
        Nr = 0;       % number of successful times
        for k=1:c
            for s = 1:sum(TestSet(k,:)~=0)
                te = 0;
                faceTest = imread(strcat('.\Cropped Yale\yaleB',num2str(k),'.\',num2str(TestSet(k,s)),'.pgm')); %read test image
                faceTestV = faceTest(:); %vector of test image
                b = 2*double(faceTestV)/255-1;
                
                b = W*b;
                b = b/norm(b);
                
                q = invG*b;
                
                %%%%%%%%%%%%%%%%%%%%Projection-Neural-Network-Based Algorithm%%%%%%%%%%
                T = [0];
                t = T;
                X = zeros(c*ci,1); % initial point
                Y = zeros(c*ci,1);
                x = X;
                Y = x - min(max(x+Y, -1), 1);
                y = Y;
                temp = 0;
%                 while te<0.5
                    while sum(x~=0)<K(sparse)
                    %                     while norm(P*x-q)>1e-1
                    %                                 while norm(A*x-b)/norm(A)>1*10^-2
                    %                     for t = 1:500
                    t = t+1;
                    %                     sum(x~=0);
                    tmp=P*x-q+0.01*min(max(x+y, -1), 1);
                    x=x-0.3*tmp;  %based on L1-norm
                    %                     IPgz = IP*min(max(z,-1),1);
                    %                     temp = temp-IPgz;
                    %                     z = temp-IPgz+min(max(z,-1),1)+q;  %based on L1-norm
                    %                     z = (I-P)*x+P*g(z)+q;
                    y=min(max(x+y, -1), 1);
                    
                    %                     T = [T t];
                    %                     Z = [Z z];
                    %                     X = [X x];
%                     te = toc(t0);     %end of the time
                end
                
                %                         figure (1)
                %                         plot(T,Z)
                %                         box on
                %                         xlabel('time (sec)')
                %                         ylabel('state vector z')
                %
                %                         figure (2)
                %                         plot(T,X)
                %                         box on
                %                         xlabel('time (sec)')
                %                         ylabel('output vector x')
                
                %%%%%%%%%%%%%%%%%%%%%%%Classify%%%%%%%%%%%%%%%%%%%%%
                Delta = [];
                for i=1:c
                    delta = x((i-1)*ci+1:i*ci);
                    dist = norm(A(:,(i-1)*ci+1:i*ci)*delta-b)/(norm(delta)*sum(delta~=0));
                    Delta = [Delta;dist];
                end
                [value,v] = min(Delta);      %class of the test image
                
                if v==k
                    Nr = Nr+1;
                end
                
            end
            k
        end
        te = toc(t0);     %end of the time
        
        Rrate = Nr/(sum(count')-c*ci)*100; %computing recognition rate
        Rate(sparse,Downrate) = Rrate;
        
        %     disp(['For m=' num2str(m) ', K=' num2str(K)]);
        
        fprintf(1,'%s%u\t%s%u\t\t%f\n', 'm=', m, 'K=', K(sparse), Rrate);
        pause(0.2) ;
    end
    disp('------------------------');
end

filen = sprintf('%s%s.mat','RecRate',strrep(num2str(fix(clock)),' ',''));
save(filen,'Rate')

toc     %end of the time
