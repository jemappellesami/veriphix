OPENQASM 2.0;
include "qelib1.inc";
qreg q400[3];
rx(pi) q400[2];
rz(5*pi/4) q400[2];
cx q400[2],q400[1];
cx q400[0],q400[1];
