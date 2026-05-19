OPENQASM 2.0;
include "qelib1.inc";
qreg q24[3];
rx(pi/2) q24[0];
rx(3*pi/4) q24[2];
rz(pi) q24[2];
cx q24[1],q24[2];
cx q24[1],q24[0];
