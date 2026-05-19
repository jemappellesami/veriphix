OPENQASM 2.0;
include "qelib1.inc";
qreg q160[4];
rx(5*pi/4) q160[3];
rz(pi/4) q160[3];
cx q160[2],q160[3];
cx q160[2],q160[1];
cx q160[1],q160[0];
