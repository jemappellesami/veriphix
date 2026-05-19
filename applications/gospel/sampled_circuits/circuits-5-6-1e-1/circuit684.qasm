OPENQASM 2.0;
include "qelib1.inc";
qreg q685[5];
cx q685[4],q685[3];
cx q685[3],q685[2];
cx q685[2],q685[1];
cx q685[0],q685[1];
rx(pi/4) q685[1];
