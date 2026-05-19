OPENQASM 2.0;
include "qelib1.inc";
qreg q507[3];
rx(pi) q507[0];
cx q507[0],q507[1];
cx q507[1],q507[2];
rx(pi/4) q507[0];
