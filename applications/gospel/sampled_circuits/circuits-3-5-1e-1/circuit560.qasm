OPENQASM 2.0;
include "qelib1.inc";
qreg q561[3];
rx(pi/4) q561[0];
cx q561[1],q561[0];
cx q561[1],q561[2];
rx(pi/4) q561[0];
