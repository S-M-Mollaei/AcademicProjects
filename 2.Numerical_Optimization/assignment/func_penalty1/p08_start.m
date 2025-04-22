function x = p08_start ( n )

%*****************************************************************************80
%
%% p08_start returns a starting point for optimization for problem 8.
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
%    integer N, the number of variables X.
%
%  Output:
%
%    real X(N), a starting point for the optimization.
%
  x = ( 1 : n )';

  return
end
