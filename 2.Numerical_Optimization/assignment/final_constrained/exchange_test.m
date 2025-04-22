clc
clear all

s = load('mytest1.mat')

alpha=s.alpha;
kmax=s.kmax;
tolgrad=s.tolgrad;

f=s.f;
x0=s.x0;

kmax = 1000;

n = 10^5;
x0 = zeros(n,1);
x0(:) = 6;
k_h = 0;

f = @(x) (jong_f ( n, x ));

save('jong_diff_test(10^5).mat','alpha','f','kmax','tolgrad','x0', 'n')