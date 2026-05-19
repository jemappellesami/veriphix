OPENQASM 2.0;
include "qelib1.inc";
qreg q997[4];
rx(pi/4) q997[3];
rz(7*pi/4) q997[3];
rx(pi/2) q997[3];
cx q997[2],q997[3];
cx q997[2],q997[1];
