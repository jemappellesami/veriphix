OPENQASM 2.0;
include "qelib1.inc";
qreg q91[5];
cx q91[2],q91[1];
rz(7*pi/4) q91[4];
cx q91[1],q91[0];
cx q91[3],q91[4];
cx q91[3],q91[2];
