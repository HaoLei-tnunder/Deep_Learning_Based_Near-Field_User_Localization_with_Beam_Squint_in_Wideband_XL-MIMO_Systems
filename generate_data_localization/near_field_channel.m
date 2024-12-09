function [H ] = near_field_channel(Nt, d, fc, B, M, r, theta)

H = zeros(  M,Nt);

c = 3e8;

% f = zeros(1, M);
nn = -(Nt-1)/2:1:(Nt-1)/2;

r0 = sqrt(  r^2 + (nn*d).^2 - 2*r*nn*d*sin(theta)   );

f = fc + B * ((2 * (1:M) / (M - 1)) - 1) / 2;

Rmin = 5;
Rmax = 50;
user_theta_max =    60/180*pi;
user_theta_min =   -60/180*pi;

r_NLoS = Rmin + rand(1, 2) * (Rmax - Rmin);
theta_NLoS = user_theta_min + rand(1, 2) * (user_theta_max - user_theta_min);

r_NLoS0 = sqrt(  r_NLoS(1)^2 + (nn*d).^2 - 2*r_NLoS(1)*nn*d*sin(theta_NLoS(1))   );
r_NLoS1 = sqrt(  r_NLoS(2)^2 + (nn*d).^2 - 2*r_NLoS(2)*nn*d*sin(theta_NLoS(2))   );

deltaTheta = theta_NLoS - theta;
distance1 =  sqrt( r^2 + r_NLoS(1)^2 - 2 * r * r_NLoS(1) * cos(deltaTheta(1)));
distance2 =  sqrt( r^2 + r_NLoS(2)^2 - 2 * r * r_NLoS(2) * cos(deltaTheta(2)));

beta =1/2*  (randn(1, 2) + 1j*randn(1, 2))/sqrt(2);

p1 = zeros(1,Nt);
n_k_min = max(1,  floor( rand * 4 ));
n_k_max = ceil (n_k_min +  rand * ( 4- n_k_min));
%     n_k_min = 1;
%     n_k_max = 3;    
p1(  1,   (n_k_min * 128 -127) :  (n_k_max*128) ) = ones( 1,  (n_k_max*128) - (n_k_min * 128 -127) +1);

p2 = zeros(1,Nt);
n_k_min = max(1,  floor( rand * 4 ));
n_k_max = ceil (n_k_min +  rand * ( 4- n_k_min));
p2(  1,   (n_k_min * 128 -127) :  (n_k_max*128) ) = ones( 1,  (n_k_max*128) - (n_k_min * 128 -127) +1);


p3 = zeros(1,Nt);
n_k_min = max(1,  floor( rand * 4 ));
n_k_max = ceil (n_k_min +  rand * ( 4- n_k_min));
p3(  1,   (n_k_min * 128 -127) :  (n_k_max*128) ) = ones( 1,  (n_k_max*128) - (n_k_min * 128 -127) +1);
values = rand(1,Nt);
p3 = p3.*values;

for m = 1:M
%          f(m)=fc+ (m-1) * B/M;
%    f(m)=fc+B/(2)*(2*m/(M-1)-1);
  
%    beta = c / (4 * pi * f(m) * r);   

%    H(m,:) = f(m)/fc*  exp(-1j*2*pi*f(m)*r0/c) ;
%    H(m,:) =  exp(-1j*2*pi*f(m)*r0/c) ;
    H_LoS =  1/r* exp(-1j*2*pi*f(m)*r0/c) .* p1;   

    H_NLoS1 = beta(1) * 1/ r_NLoS(1) * exp(-1j*2*pi*f(m)*r_NLoS0/c)  * 1 / distance1 *  exp(-1j*2*pi*f(m)*distance1/c) .* p2 ;
% 
    H_NLoS2 = beta(2) * 1/ r_NLoS(2) * exp(-1j*2*pi*f(m)*r_NLoS1/c)  * 1 / distance2 *  exp(-1j*2*pi*f(m)*distance2/c) .* p3;
   
%     H(m,:) = H_LoS  ;
    H(m,:) = H_LoS  +  H_NLoS1 + H_NLoS2;

%    H(m,:) = c / (4 * pi * f(m) * r) *  exp(-1j*2*pi*f(m)*r0/c) ;


end


% beta =  (randn(1, 1) + 1j*randn(1, 1))/sqrt(2);
% 
% H = beta .*   H ;

end

