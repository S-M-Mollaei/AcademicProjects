function [ know, x ] = p08_sol ( n )

%*****************************************************************************80
%
%% p08_sol returns the solution for problem 8.
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
%    integer N, the order of the problem.  This value
%    is only needed for those problems with variable N.
%
%  Output:
%
%    integer KNOW.
%    If KNOW is 0, then the solution is not known.
%    If KNOW is positive, then the solution is known, and is returned in X.
%
%    real X(N), the solution, if known.
%
  know = 0;

  x = zeros ( n, 1 );

  return
end
