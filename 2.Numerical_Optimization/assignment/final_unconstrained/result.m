
%% LOADING THE VARIABLES FOR THE TEST FOR X0 = [1.2;1.2]

clear
close all
clc

c1 = 1e-4;
rho = 0.3;
btmax = 50;

disp('++++++++++++++++++++++++++ MAIN FUNCTION ++++++++++++++++++++++++++++++++')
load('mytest1.mat')
disp('///////////////////////// FIRST POINT ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0 = [1.2;1.2]

disp('**** STEEPEST DESCENT: START WITH X0=[1.2;1.2] *****')
tic
[xk1, fk1, gradfk_norm1, k1, xseq1, btseq1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk1), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq1))])
disp(['grad_norm: ', mat2str(gradfk_norm1)])
disp('***************** DONE *******************')


%% RUN THE NEWTON FOR X0 = [1.2;1.2]

disp('**** NEWTON: START WITH X0=[1.2;1.2] *****')
tic
[xk_n1, fk_n1, gradfk_norm_n1, k_n1, xseq_n1, btseq_n1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk_n1), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_n1))])
disp(['grad_norm: ', mat2str(gradfk_norm_n1)])
disp('****************** DONE ******************')



%% LOADING THE VARIABLES FOR THE TEST FOR X0 = [-1.2;1]

load('mytest2.mat')
disp('///////////////////////// SECOND POINT ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0 = [-1.2;1]

disp('**** STEEPEST DESCENT: START WITH X0=[-1.2;1] *****')
tic
[xk2, fk2, gradfk_norm2, k2, xseq2, btseq2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk2), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq2))])
disp(['grad_norm: ', mat2str(gradfk_norm2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0 = [-1.2;1]

disp('**** NEWTON: START WITH X0=[-1.2;1] *****')
tic
[xk_n2, fk_n2, gradfk_norm_n2, k_n2, xseq_n2, btseq_n2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0)])
disp(['xk: ', mat2str(xk_n2), ' (actual minimum: [1; 1]);'])
disp(['f(xk): ', num2str(fk_n2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_n2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_n2))])
disp(['grad_norm: ', mat2str(gradfk_norm_n2)])
disp('****************** DONE ******************')

%% LOADING THE VARIABLES FOR THE TEST FOR X0_Rosenbrock = 1/3
disp('++++++++++++++++++++++++++ EXTENDED ROSENBROCK VALLEY TEST FUNCTION ++++++++++++++++++++++++++++++++')
load('ExRosenbrockValley_test(.33).mat')
disp('///////////////////////// POINT 0.33 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Rosenbrock = 1/3

disp('**** STEEPEST DESCENT: START WITH X0_Rosenbrock = 1/3 *****')
tic
[xkr, fkr, gradfk_normr, kr, xseqr, btseqr] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkr)), ', xk_max: ', mat2str(max(xkr))])
disp(['f(xk): ', num2str(fkr), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kr),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqr))])
disp(['grad_norm: ', mat2str(gradfk_normr)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Rosenbrock = 1/3

disp('**** NEWTON: START WITH X0_Rosenbrock = 1/3 *****')
tic
[xk_nr, fk_nr, gradfk_norm_nr, k_nr, xseq_nr, btseq_nr] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nr)), ', xk_max: ', mat2str(max(xk_nr))])
disp(['f(xk): ', num2str(fk_nr), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nr),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nr))])
disp(['grad_norm: ', mat2str(gradfk_norm_nr)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_Rosenbrock = 1

load('ExRosenbrockValley_test(1).mat')
disp('///////////////////////// POINT 1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Rosenbrock = 1

disp('**** STEEPEST DESCENT: START WITH X0_Rosenbrock = 1 *****')
tic
[xkr1, fkr1, gradfk_normr1, kr1, xseqr1, btseqr1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkr1)), ', xk_max: ', mat2str(max(xkr1))])
disp(['f(xk): ', num2str(fkr1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kr1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqr1))])
disp(['grad_norm: ', mat2str(gradfk_normr1)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Rosenbrock = 1

disp('**** NEWTON: START WITH X0_Rosenbrock = 1 *****')
tic
[xk_nr1, fk_nr1, gradfk_norm_nr1, k_nr1, xseq_nr1, btseq_nr1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nr1)), ', xk_max: ', mat2str(max(xk_nr1))])
disp(['f(xk): ', num2str(fk_nr1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nr1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nr1))])
disp(['grad_norm: ', mat2str(gradfk_norm_nr1)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_Rosenbrock = -1

load('ExRosenbrockValley_test(-1).mat')
disp('///////////////////////// POINT -1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Rosenbrock = -1

disp('**** STEEPEST DESCENT: START WITH X0_Rosenbrock = -1 *****')
tic
[xkr2, fkr2, gradfk_normr2, kr2, xseqr2, btseqr2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkr2)), ', xk_max: ', mat2str(max(xkr2))])
disp(['f(xk): ', num2str(fkr2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kr2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqr2))])
disp(['grad_norm: ', mat2str(gradfk_normr2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Rosenbrock = -1

disp('**** NEWTON: START WITH X0_Rosenbrock = -1 *****')
tic
[xk_nr2, fk_nr2, gradfk_norm_nr2, k_nr2, xseq_nr2, btseq_nr2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nr2)), ', xk_max: ', mat2str(max(xk_nr2))])
disp(['f(xk): ', num2str(fk_nr2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nr2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nr2))])
disp(['grad_norm: ', mat2str(gradfk_norm_nr2)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_pt1 = 1/3
disp('++++++++++++++++++++++++++ PENALTY TEST FUNCTION ++++++++++++++++++++++++++++++++')
load('penaltyfunc1_test(.33).mat')
disp('///////////////////////// POINT 0.33 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_pt1 = 1/3

disp('**** STEEPEST DESCENT: START WITH X0_pt1 = 1/3 *****')
tic
[xkp, fkp, gradfk_normp, kp, xseqp, btseqp] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkp)), ', xk_max: ', mat2str(max(xkp))])
disp(['f(xk): ', num2str(fkp), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kp),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqp))])
disp(['grad_norm: ', mat2str(gradfk_normp)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_pt1 = 1/3

disp('**** NEWTON: START WITH X0_pt1 = 1/3 *****')
tic
[xk_np, fk_np, gradfk_norm_np, k_np, xseq_np, btseq_np] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_np)), ', xk_max: ', mat2str(max(xk_np))])
disp(['f(xk): ', num2str(fk_np), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_np),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_np))])
disp(['grad_norm: ', mat2str(gradfk_norm_np)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_pt1 = 1

load('penaltyfunc1_test(1).mat')
disp('///////////////////////// POINT 1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_pt1 = 1

disp('**** STEEPEST DESCENT: START WITH X0_pt1 = 1 *****')
tic
[xkp1, fkp1, gradfk_normp1, kp1, xseqp1, btseqp1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkp1)), ', xk_max: ', mat2str(max(xkp1))])
disp(['f(xk): ', num2str(fkp1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kp1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqp1))])
disp(['grad_norm: ', mat2str(gradfk_normp1)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_pt1 = 1

disp('**** NEWTON: START WITH X0_pt1 = 1 *****')
tic
[xk_np1, fk_np1, gradfk_norm_np1, k_np1, xseq_np1, btseq_np1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_np1)), ', xk_max: ', mat2str(max(xk_np1))])
disp(['f(xk): ', num2str(fk_np1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_np1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_np1))])
disp(['grad_norm: ', mat2str(gradfk_norm_np1)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_pt1 = -1

load('penaltyfunc1_test(-1).mat')
disp('///////////////////////// POINT -1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_pt1 = -1

disp('**** STEEPEST DESCENT: START WITH X0_pt1 = -1 *****')
tic
[xkp2, fkp2, gradfk_normp2, kp2, xseqp1, btseqp2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkp2)), ', xk_max: ', mat2str(max(xkp2))])
disp(['f(xk): ', num2str(fkp2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kp2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqp2))])
disp(['grad_norm: ', mat2str(gradfk_normp2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_pt1 = -1

disp('**** NEWTON: START WITH X0_pt1 = -1 *****')
tic
[xk_np2, fk_np2, gradfk_norm_np2, k_np2, xseq_np2, btseq_np2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_np2)), ', xk_max: ', mat2str(max(xk_np2))])
disp(['f(xk): ', num2str(fk_np2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_np2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_np2))])
disp(['grad_norm: ', mat2str(gradfk_norm_np2)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_trigonometric = 1/3
disp('++++++++++++++++++++++++++ TRIGONOMETRIC TEST FUNCTION ++++++++++++++++++++++++++++++++')
load('trigonometric_test(.33).mat')
disp('///////////////////////// POINT 0.33 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_trigonometric = 1/3

disp('**** STEEPEST DESCENT: START WITH X0_trigonometric = 1/3 *****')
tic
[xkt, fkt, gradfk_normt, kt, xseqt, btseqt] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkt)), ', xk_max: ', mat2str(max(xkt))])
disp(['f(xk): ', num2str(fkt), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kt),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqt))])
disp(['grad_norm: ', mat2str(gradfk_normt)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_trigonometric = 1/3

disp('**** NEWTON: START WITH X0_trigonometric = 1/3 *****')
tic
[xk_nt, fk_nt, gradfk_norm_nt, k_nt, xseq_nt, btseq_nt] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nt)), ', xk_max: ', mat2str(max(xk_nt))])
disp(['f(xk): ', num2str(fk_nt), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nt),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nt))])
disp(['grad_norm: ', mat2str(gradfk_norm_nt)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_trigonometric = 1

load('trigonometric_test(1).mat')
disp('///////////////////////// POINT 1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_trigonometric = 1

disp('**** STEEPEST DESCENT: START WITH X0_trigonometric = 1 *****')
tic
[xkt1, fkt1, gradfk_normt1, kt1, xseqt1, btseqt1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkt1)), ', xk_max: ', mat2str(max(xkt1))])
disp(['f(xk): ', num2str(fkt1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kt1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqt1))])
disp(['grad_norm: ', mat2str(gradfk_normt1)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_trigonometric = 1

disp('**** NEWTON: START WITH X0_trigonometric = 1 *****')
tic
[xk_nt1, fk_nt1, gradfk_norm_nt1, k_nt1, xseq_nt1, btseq_nt1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nt1)), ', xk_max: ', mat2str(max(xk_nt1))])
disp(['f(xk): ', num2str(fk_nt1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nt1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nt1))])
disp(['grad_norm: ', mat2str(gradfk_norm_nt1)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_trigonometric = -1

load('trigonometric_test(-1).mat')
disp('///////////////////////// POINT -1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_trigonometric = -1

disp('**** STEEPEST DESCENT: START WITH X0_trigonometric = -1 *****')
tic
[xkt2, fkt2, gradfk_normt2, kt2, xseqt2, btseqt2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkt2)), ', xk_max: ', mat2str(max(xkt2))])
disp(['f(xk): ', num2str(fkt2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kt2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqt2))])
disp(['grad_norm: ', mat2str(gradfk_normt2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_trigonometric = -1

disp('**** NEWTON: START WITH X0_trigonometric = -1 *****')
tic
[xk_nt2, fk_nt2, gradfk_norm_nt2, k_nt2, xseq_nt2, btseq_nt2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nt2)), ', xk_max: ', mat2str(max(xk_nt2))])
disp(['f(xk): ', num2str(fk_nt2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nt2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nt2))])
disp(['grad_norm: ', mat2str(gradfk_norm_nt2)])
disp('****************** DONE ******************')

%% LOADING THE VARIABLES FOR THE TEST FOR X0_hilbert = 1/3
disp('++++++++++++++++++++++++++ HILBERT TEST FUNCTION ++++++++++++++++++++++++++++++++')
load('hilbert_test(.33).mat')
disp('///////////////////////// POINT 0.33 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_hilbert = 1/3

disp('**** STEEPEST DESCENT: START WITH X0_hilbert = 1/3 *****')
tic
[xkh, fkh, gradfk_normh, kh, xseqh, btseqh] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkh)), ', xk_max: ', mat2str(max(xkh))])
disp(['f(xk): ', num2str(fkh), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kh),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqh))])
disp(['grad_norm: ', mat2str(gradfk_normh)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_hilbert = 1/3

disp('**** NEWTON: START WITH X0_hilbert = 1/3 *****')
tic
[xk_nh, fk_nh, gradfk_norm_nh, k_nh, xseq_nh, btseq_nh] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nh)), ', xk_max: ', mat2str(max(xk_nh))])
disp(['f(xk): ', num2str(fk_nh), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nh),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nh))])
disp(['grad_norm: ', mat2str(gradfk_norm_nh)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_hilbert = 1

load('hilbert_test(1).mat')
disp('///////////////////////// POINT 1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_hilbert = 1

disp('**** STEEPEST DESCENT: START WITH X0_hilbert = 1 *****')
tic
[xkh1, fkh1, gradfk_normh1, kh1, xseqh1, btseqh1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkh1)), ', xk_max: ', mat2str(max(xkh1))])
disp(['f(xk): ', num2str(fkh1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kh1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqh1))])
disp(['grad_norm: ', mat2str(gradfk_normh1)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_hilbert = 1

disp('**** NEWTON: START WITH X0_hilbert = 1 *****')
tic
[xk_nh1, fk_nh1, gradfk_norm_nh1, k_nh1, xseq_nh1, btseq_nh1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nh1)), ', xk_max: ', mat2str(max(xk_nh1))])
disp(['f(xk): ', num2str(fk_nh1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nh1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nh1))])
disp(['grad_norm: ', mat2str(gradfk_norm_nh1)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_hilbert = -1

load('hilbert_test(-1).mat')
disp('///////////////////////// POINT -1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_hilbert = -1

disp('**** STEEPEST DESCENT: START WITH X0_hilbert = -1 *****')
tic
[xkh2, fkh2, gradfk_normh2, kh2, xseqh2, btseqh2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkh2)), ', xk_max: ', mat2str(max(xkh2))])
disp(['f(xk): ', num2str(fkh2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kh2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqh2))])
disp(['grad_norm: ', mat2str(gradfk_normh2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_hilbert = -1

disp('**** NEWTON: START WITH X0_hilbert = -1 *****')
tic
[xk_nh2, fk_nh2, gradfk_norm_nh2, k_nh2, xseq_nh2, btseq_nh2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_nh2)), ', xk_max: ', mat2str(max(xk_nh2))])
disp(['f(xk): ', num2str(fk_nh2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_nh2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_nh2))])
disp(['grad_norm: ', mat2str(gradfk_norm_nh2)])
disp('****************** DONE ******************')

%% LOADING THE VARIABLES FOR THE TEST FOR X0_Gregory = 1/3
disp('++++++++++++++++++++++++++ Gregory@Karney Tridiagonal Matrix TEST FUNCTION ++++++++++++++++++++++++++++++++')
load('Gregory_test(.33).mat')
disp('///////////////////////// POINT 0.33 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Gregory = 1/3

disp('**** STEEPEST DESCENT: START WITH X0_Gregory = 1/3 *****')
tic
[xkg, fkg, gradfk_normg, kg, xseqg, btseqg] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkg)), ', xk_max: ', mat2str(max(xkg))])
disp(['f(xk): ', num2str(fkg), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kg),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqg))])
disp(['grad_norm: ', mat2str(gradfk_normg)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Gregory = 1/3

disp('**** NEWTON: START WITH X0_Gregory = 1/3 *****')
tic
[xk_ng, fk_ng, gradfk_norm_ng, k_ng, xseq_ng, btseq_ng] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_ng)), ', xk_max: ', mat2str(max(xk_ng))])
disp(['f(xk): ', num2str(fk_ng), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_ng),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_ng))])
disp(['grad_norm: ', mat2str(gradfk_norm_ng)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_Gregory = 1

load('Gregory_test(1).mat')
disp('///////////////////////// POINT 1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Gregory = 1

disp('**** STEEPEST DESCENT: START WITH X0_Gregory = 1 *****')
tic
[xkg1, fkg1, gradfk_normg1, kg1, xseqg1, btseqg1] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkg1)), ', xk_max: ', mat2str(max(xkg1))])
disp(['f(xk): ', num2str(fkg1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kg1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqg1))])
disp(['grad_norm: ', mat2str(gradfk_normg1)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Gregory = 1

disp('**** NEWTON: START WITH X0_Gregory = 1 *****')
tic
[xk_ng1, fk_ng1, gradfk_norm_ng1, k_ng1, xseq_ng1, btseq_ng1] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_ng1)), ', xk_max: ', mat2str(max(xk_ng1))])
disp(['f(xk): ', num2str(fk_ng1), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_ng1),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_ng1))])
disp(['grad_norm: ', mat2str(gradfk_norm_ng1)])
disp('****************** DONE ******************')


%% LOADING THE VARIABLES FOR THE TEST FOR X0_Gregory = -1

load('Gregory_test(-1).mat')
disp('///////////////////////// POINT -1 ////////////////////////////')
%% RUN THE STEEPEST DESCENT FOR X0_Gregory = -1

disp('**** STEEPEST DESCENT: START WITH X0_Gregory = -1 *****')
tic
[xkg2, fkg2, gradfk_normg2, kg2, xseqg2, btseqg2] = ...
    steepest_desc_bcktrck(x0, f, gradf, alpha, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** STEEPEST DESCENT: FINISHED *****')
disp('**** STEEPEST DESCENT: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xkg2)), ', xk_max: ', mat2str(max(xkg2))])
disp(['f(xk): ', num2str(fkg2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(kg2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseqg2))])
disp(['grad_norm: ', mat2str(gradfk_normg2)])
disp('****************** DONE ******************')


%% RUN THE NEWTON FOR X0_Gregory = -1

disp('**** NEWTON: START WITH X0_Gregory = -1 *****')
tic
[xk_ng2, fk_ng2, gradfk_norm_ng2, k_ng2, xseq_ng2, btseq_ng2] = ...
    newton_bcktrck(x0, f, gradf, Hessf, kmax, ...
    tolgrad, c1, rho, btmax);
toc
disp('**** NEWTON: FINISHED *****')
disp('**** NEWTON: RESULTS *****')
disp('-------------------------------------')
disp(['START POINT: ', mat2str(x0(1))])
disp(['xk_min: ', mat2str(min(xk_ng2)), ', xk_max: ', mat2str(max(xk_ng2))])
disp(['f(xk): ', num2str(fk_ng2), ' (actual min. value: 0);'])
disp(['N. of Iterations: ', num2str(k_ng2),'/',num2str(kmax), ';'])
disp(['BackTrackMax: ', mat2str(max(btseq_ng2))])
disp(['grad_norm: ', mat2str(gradfk_norm_ng2)])
disp('****************** DONE ******************')
