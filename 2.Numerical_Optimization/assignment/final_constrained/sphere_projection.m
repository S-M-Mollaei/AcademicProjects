function xhat = sphere_projection(x, c, r)
% 
% function xhat = sphere_projection(x, c, r)
%
% Function that performs the projection of a vector on the boundaries of an
% n-dimensional sphere.
%
% INPUTS:
% x = n-dimensional vector;
% c = n-dimensional vector that is the center of teh sphere
% r = radius of the sphere;
%
% OUTPUTS:
% xhat = it is x if x is in the sphere, otherwise it is the 
% projection of x on the boundary of the sphere.
%

xc_dist = norm(x-c);
if xc_dist > r
    xc_versor = (x - c)/xc_dist;
    xhat = c + xc_versor * r;
else
    xhat = x;
end

end