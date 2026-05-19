OPENQASM 2.0;
include "qelib1.inc";
qreg q14[3];
rx(pi/4) q14[2];
cx q14[1],q14[2];
cx q14[1],q14[0];
rx(pi/4) q14[1];
