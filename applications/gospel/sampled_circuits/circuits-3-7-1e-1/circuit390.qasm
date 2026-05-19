OPENQASM 2.0;
include "qelib1.inc";
qreg q391[3];
cx q391[1],q391[2];
rx(5*pi/4) q391[1];
rz(pi) q391[2];
cx q391[2],q391[1];
cx q391[0],q391[1];
