function [ r_hat_avg  , theta_hat_avg , y1, y2 ] = CBS_Low_Localization_Scheme( h,  Nt, M, K,B, d, f,   SNR_linear,   Rmin,    Rmax,   user_theta_max ,  user_theta_min ,theta,r)


%%   angle
T=4;
y = zeros(K,M);
y1 = zeros(K,M,T);

y_abs_1= zeros(K,M,T);
theta_index = zeros(T,1);
% theta_index1 = zeros(T,1);
theta_hat = zeros(T,1);
% theta_hat1 = zeros(T,1);
r0 = 5  ;   %start
theta0 = user_theta_max ;
rc = 5 ;   %end
thetac = user_theta_min   ;
 [  theta_M  ,  ~ ] = Beam_Squint_trajectory(B, M, f  ,   theta0, r0,  thetac,  rc );
w = generate_beamfoucing_vector( Nt, M,B, d, f, r0, theta0 , rc, thetac, 0 );
for m = 1 : M
    y(1,m) =  conj(h(m,:) )  * w(:,m)    ;
end
power = (abs(y(1,:)) .^2);
sigma2 = power / SNR_linear;
n_l = sqrt(sigma2);
for t = 1 : T

    noise = n_l .* sqrt(1 / 2) .*(   randn(1,M ) + 1i * randn(1,M )  );

    y1(1,:,t) = y(1,:) + noise;

%     y1(1,:,t) = y(1,:) ;

    %     for m = 1 : M
    % %         y(1,m,t) = y(1,m,t) * 4 * pi * f(m) /c ;
    %     end

    y_abs_1(1,:,t) = abs(y1(1,:,t));
    y_abs_1(1,:,t) = y_abs_1(1,:,t)/max(y_abs_1(1,:,t));

%     [max_values, max_indices] = maxk(y_abs_1(1,:,t), 2);
% 
%     if max_values(2) > 0.8
%         theta_index(t) =round( sum(max_indices) / 2   );
%     else
%         [~,theta_index(t)] = max(y_abs_1(1,:,t));
%     end

    [~,theta_index(t)] = max(y_abs_1(1,:,t));
   
    theta_hat(t) = theta_M(theta_index(t));

%     [~,theta_index1(t)] = max(y_abs_1(1,:,t));
%     theta_hat1(t) = theta_M(theta_index1(t));    

end

theta_hat_avg = sum(theta_hat)/T;


%%  distance


T=2;
y = zeros(K,M);
y2 = zeros(K,M,T);
y_abs_2= zeros(K,M,T);
r_hat = zeros(T,1);
r_index = zeros(T,1);
r0 = Rmax  ;   %start
% theta0 = theta_hat_avg;
theta0 = theta;

rc = Rmin ;   %end
thetac = theta0  ;

w = generate_beamfoucing_vector( Nt, M,B, d, f, r0, theta0 , rc, thetac , 0 );
[  ~  ,  r_M ] = Beam_Squint_trajectory(B, M, f  ,   theta0, r0,  thetac,  rc );
for m = 1 : M
    y(1,m) =  conj(h(m,:) )  * w(:,m)    ;
end
power = (abs(y) .^2);
sigma2 = power / SNR_linear;

n_l = sqrt(sigma2);

for t = 1 : T


    noise = n_l .* sqrt(1 / 2) .*(   randn(1,M ) + 1i * randn(1,M )  );
    y2(1,:,t) = y+ noise;
%     y2(1,:,t) = y;

    %     for m = 1 : M
    % %         y(1,m,t) = y(1,m,t) * 4 * pi * f(m) /c ;
    %     end

    y_abs_2(1,:,t) = abs(y2(1,:,t));
    y_abs_2(1,:,t) = y_abs_2(1,:,t)/max(y_abs_2(1,:,t));
    [~,r_index(t)] = max(y_abs_2(1,:,t));

    r_hat(t) = r_M(r_index(t));

end

r_hat_avg = sum(r_hat)/T;
%  r_hat_avg = 0;




end

