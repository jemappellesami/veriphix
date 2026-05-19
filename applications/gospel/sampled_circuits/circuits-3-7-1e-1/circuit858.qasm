OPENQASM 2.0;
include "qelib1.inc";
qreg q859[3];
rx(7*pi/4) q859[2];
rz(pi/4) q859[2];
rx(7*pi/4) q859[2];
cx q859[2],q859[1];
cx q859[0],q859[1];
