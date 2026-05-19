OPENQASM 2.0;
include "qelib1.inc";
qreg q982[3];
rx(pi/2) q982[2];
cx q982[2],q982[1];
cx q982[0],q982[1];
