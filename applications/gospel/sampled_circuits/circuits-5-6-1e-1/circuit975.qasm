OPENQASM 2.0;
include "qelib1.inc";
qreg q976[5];
cx q976[4],q976[3];
cx q976[2],q976[3];
cx q976[1],q976[2];
cx q976[0],q976[1];
rx(pi/4) q976[1];
