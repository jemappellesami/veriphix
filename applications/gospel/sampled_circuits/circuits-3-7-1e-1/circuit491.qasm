OPENQASM 2.0;
include "qelib1.inc";
qreg q492[3];
cx q492[2],q492[1];
rx(pi/4) q492[2];
cx q492[2],q492[1];
cx q492[0],q492[1];
