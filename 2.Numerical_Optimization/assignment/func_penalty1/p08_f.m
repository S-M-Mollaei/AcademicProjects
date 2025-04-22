function f = p08_f ( n, x )

%*****************************************************************************80
%
%% p08_f evaluates the objective function for problem 8.
%
%  Licensing:
%
%    This code is distributed under the GNU LGPL license.
%
%  Modified:
%
%    16 October 2011
%
%  Author:
%
%    John Burkardt
%
%  Input:
%
%    integer N, the number of variables.
%
%    real X(N), the argument of the objective function.
%
%  Output:
%
%    real F, the value of the objective function.
%
  ap = 0.00001;

  t1 = - 0.25 + sum ( x(1:n).^2 );

  t2 = sum ( ( x(1:n) - 1.0 ).^2 );

  f = ap * t2 + t1 * t1;
  
  f = 0.00001 * sum ( ( x(1:n) - 1.0 ).^2 ) + (- 0.25 + sum ( x(1:n).^2 )) * (- 0.25 + sum ( x(1:n).^2 ));

  return
end
