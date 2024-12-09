function  w = generate_beamfoucing_vector( Nt,M, B,d, f, r0, theta0 , rc, thetac , mood )
c = 3e8;
if mood == 1   %the phase-shifters (PSs) based beamforming

    w = zeros(Nt,M);
    nn = -(Nt-1)/2:1:(Nt-1)/2;
    rr = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));   %start
    phi = zeros(Nt,1);
    for n = 1 : Nt
        for m = 1 : M
            phi(n)= f(1)/c*rr(n);
            w(n,m)=1/sqrt(Nt)*exp(-1j*2*pi*phi(n));
        end
    end


elseif mood == 0  %  time-delay lines (TDs)  +  phase-shifters (PSs)

    w = zeros(Nt,M);

    nn = -(Nt-1)/2:1:(Nt-1)/2;

    rr = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));   %start
    rrc = sqrt(rc^2 + (nn*d).^2 - 2*rc*nn*d*sin(thetac));   %end

    phi = zeros(Nt,1);
    t = zeros(Nt,1);

    for n = 1 : Nt
        for m = 1 : M
            phi(n)= f(1)/c*rr(n);
            t(n) = f(M)/B/c*rrc(n) - phi(n)/B;
            w(n,m)=1/sqrt(Nt)*exp(-1j*2*pi*phi(n))*exp(-1j*2*pi* (   +f(m) -f(1)  ) * t(n))  ;
        end
    end


else


    w = zeros(Nt,M);

    nn = -(Nt-1)/2:1:(Nt-1)/2;


    rr = sqrt(r0^2 + (nn*d).^2 - 2*r0*nn*d*sin(theta0));   %start
    rrc = sqrt(rc^2 + (nn*d).^2 - 2*rc*nn*d*sin(thetac));   %end

    phi = zeros(Nt,1);
    t = zeros(Nt,1);


    for n = 1 : Nt
            for m = 1 : M
                phi(n)= f(1)/c*rr(n);
                t(n) = f(M)/B/c*rrc(n) - phi(n)/B;
                w(n,m)=1/sqrt(Nt)*exp(-1j*2*pi*phi(n))*exp(-1j*2*pi* (   +f(m) -f(1)  ) * t(n));
            end

    end


end

end

