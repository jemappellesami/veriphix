OPENQASM 2.0;
include "qelib1.inc";
qreg q116[3];
rx(5*pi/4) q116[1];
rz(pi/2) q116[2];
rx(pi/4) q116[2];
cx q116[2],q116[1];
cx q116[1],q116[0];
