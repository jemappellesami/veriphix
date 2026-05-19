OPENQASM 2.0;
include "qelib1.inc";
qreg q754[3];
cx q754[0],q754[1];
rx(3*pi/2) q754[2];
cx q754[1],q754[0];
cx q754[1],q754[2];
cx q754[0],q754[1];
