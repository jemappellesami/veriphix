OPENQASM 2.0;
include "qelib1.inc";
qreg q624[3];
rx(pi/4) q624[2];
rz(7*pi/4) q624[2];
cx q624[1],q624[2];
cx q624[1],q624[0];
