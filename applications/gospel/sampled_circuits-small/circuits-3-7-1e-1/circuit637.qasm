OPENQASM 2.0;
include "qelib1.inc";
qreg q638[3];
rx(5*pi/4) q638[2];
rz(pi) q638[2];
rx(3*pi/2) q638[2];
cx q638[1],q638[2];
cx q638[0],q638[1];
