OPENQASM 2.0;
include "qelib1.inc";
qreg q940[3];
rx(3*pi/4) q940[2];
rz(3*pi/4) q940[2];
cx q940[2],q940[1];
cx q940[0],q940[1];
