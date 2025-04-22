%% LOADING THE VARIABLES FOR THE TEST FOR X0 = [1.2;1.2]

clear
close all
clc

c1 = 1e-4;
rho = 0.8;
btmax = 50;

load('mytest1.mat')

%% RUN THE STEEPEST DESCENT FOR X0 = [1.2;1.2]

disp('**** STEEPEST DESCENT: START WITH X0=[1.2;1.2] *****')
tic
[xk1, fk1, gradfk_norm1, k1, xseq1, btseq1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('************************************')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk1), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq1))])
disp(['grad_norm: ', mat2str(gradfk_norm1)])
disp('************************************')
%% RUN THE NEWTON FOR X0 = [1.2;1.2]

disp('**** NEWTON: START WITH X0=[1.2;1.2] *****')
tic
[xk_n1, fk_n1, gradfk_norm_n1, k_n1, xseq_n1, btseq_n1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('************************************')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk_n1), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_n1))])
disp(['grad_norm: ', mat2str(gradfk_norm_n1)])
disp('************************************')