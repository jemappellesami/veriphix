OPENQASM 2.0;
include "qelib1.inc";
qreg q542[3];
rx(7*pi/4) q542[2];
cx q542[2],q542[1];
cx q542[1],q542[0];
rx(pi/4) q542[1];
