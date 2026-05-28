OPENQASM 2.0;
include "qelib1.inc";
qreg q928[3];
rx(7*pi/4) q928[0];
rz(5*pi/4) q928[2];
cx q928[2],q928[1];
cx q928[1],q928[0];
rx(pi/4) q928[1];
