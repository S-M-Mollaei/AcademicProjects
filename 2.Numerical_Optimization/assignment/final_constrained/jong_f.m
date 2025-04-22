function f = jong_f(n, x0)

f=0;
for i = 1:n
    f = f + i*x0(i)^2;
end
end