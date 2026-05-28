OPENQASM 2.0;
include "qelib1.inc";
qreg q255[3];
rx(pi/4) q255[0];
rz(pi) q255[0];
rx(3*pi/4) q255[2];
cx q255[2],q255[1];
cx q255[0],q255[1];
