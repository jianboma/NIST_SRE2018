function x = extract_ivector_long_and_short(statFilename, ubmFilename, tvFilename_long, tvFilename_short, ivFilename)
% extracts i-vector for stats in statFilename with UBM and T matrix in 
% ubmFilename and tvFilename, and optionally save the i-vector in ivFilename. 
%
% Inputs:
%   - statFilename : input statistics file name (string) or concatenated
%                    stats in a one-dimensional array
%   - ubmFilename  : UBM file name or a structure with UBM hyperparameters
%   - tvFilename   : total subspace file name (string) or matrix
%   - ivFilename   : output i-vector file name (optional)
%
% Outputs:
%   - x 		   : output identity vector (i-vector)  
%
% References:
%   [1] D. Matrouf, N. Scheffer, B. Fauve, J.-F. Bonastre, "A straightforward 
%       and efficient implementation of the factor analysis model for speaker 
%       verification," in Proc. INTERSPEECH, Antwerp, Belgium, Aug. 2007, 
%       pp. 1242-1245.  
%   [2] P. Kenny, "A small footprint i-vector extractor," in Proc. Odyssey, 
%       The Speaker and Language Recognition Workshop, Singapore, Jun. 2012.
%   [3] N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, and P. Ouellet, "Front-end 
%       factor analysis for speaker verification," IEEE TASLP, vol. 19, pp. 
%       788-798, May 2011. 
%
%
% Omid Sadjadi <s.omid.sadjadi@gmail.com>
% MicroSoft Research, Silicon Valley Center

if ischar(ubmFilename),
	tmp  = load(ubmFilename);
	ubm  = tmp.gmm;
elseif isstruct(ubmFilename),
	ubm = ubmFilename;
else
    error('oh dear! ubmFilename should be either a string or a structure!');
end
[ndim, nmix] = size(ubm.mu);
S = reshape(ubm.sigma, ndim * nmix, 1);
idx_sv = reshape(repmat(1 : nmix, ndim, 1), ndim * nmix, 1);

if ischar(tvFilename_long),
	tmp = load(tvFilename_long);
	T_long = tmp.T;
else
	T_long = tvFilename_long;
end
if ischar(tvFilename_short),
	tmp = load(tvFilename_short);
	T_short = tmp.T;
else
	T_short = tvFilename_short;
end
tv_dim = size(T_short, 1);
I = eye(tv_dim);
T_invS_long =  bsxfun(@rdivide, T_long, S');
T_invS_short =  bsxfun(@rdivide, T_short, S');

if ischar(statFilename)
    tmp = load(statFilename);
    N = tmp.N;
    F = tmp.F;
else
    N = statFilename(1 : nmix);
    F = statFilename(nmix + 1 : end);
end

L = I +  bsxfun(@times, T_invS_long, N(idx_sv)') * T_long' + bsxfun(@times, T_invS_short, N(idx_sv)') * T_short';
B = T_invS_long * F + T_invS_short* F ;
x = pinv(L) * B;

if ( nargin == 5)
    % create the path if it does not exist and save the file
    path = fileparts(ivFilename);
    if ( exist(path, 'dir')~=7 && ~isempty(path) ), mkdir(path); end
	parsave(ivFilename, x);
end


function parsave(fname, x) %#ok
save(fname, 'x')

return