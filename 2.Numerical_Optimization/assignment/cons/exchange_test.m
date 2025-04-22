clc
clear all

s = load('mytest1.mat')

alpha=s.alpha;
kmax=s.kmax;
tolgrad=s.tolgrad;

f=s.f;
gradf=s.gradf;
Hessf=s.Hessf;
x0=s.x0;
kmax=3000;

n = 10^4;
x0 = zeros(n,1);
x0(:) = 7;

f = @(x) (jong_f ( n, x ));
gradf = @(x) (jong_g ( n, x ));
Hessf = @(x) (jong_h ( n, x ));

save('jong_test(10^4).mat','Hessf','alpha','f','gradf','kmax','tolgrad','x0', 'n')