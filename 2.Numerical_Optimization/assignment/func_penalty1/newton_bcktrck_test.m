%% LOADING THE VARIABLES FOR THE TEST

clear 
clc

load('Gregory.mat')

c1 = 1e-4;
rho = 0.8;
btmax = 50;



%% RUN THE NEWTON ON f

disp('**** NEWTON: START f1 *****')
tic
[xk_n, fk_n, gradfk_norm_n, k_n, xseq_n, btseq_n] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('************************************')
disp(['xk: ', mat2str(xk_n), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n),'/',num2str(kmax), ';'])
disp(['btseq_n: ', mat2str(max(btseq_n))])
disp(['gradfk_norm_n: ', mat2str(gradfk_norm_n)])
disp('************************************')
disp('---')

