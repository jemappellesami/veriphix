OPENQASM 2.0;
include "qelib1.inc";
qreg q807[3];
rx(pi/4) q807[1];
rz(3*pi/2) q807[2];
cx q807[2],q807[1];
cx q807[0],q807[1];
