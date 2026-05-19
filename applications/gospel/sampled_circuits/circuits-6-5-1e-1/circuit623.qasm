OPENQASM 2.0;
include "qelib1.inc";
qreg q624[6];
cx q624[0],q624[1];
cx q624[1],q624[2];
rx(5*pi/4) q624[0];
cx q624[0],q624[1];
