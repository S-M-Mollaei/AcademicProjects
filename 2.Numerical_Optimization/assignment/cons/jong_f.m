function f = jong_f(n, x)

f=0;
for i = 1:n
    f = f + i*x(i)^2;
end
end