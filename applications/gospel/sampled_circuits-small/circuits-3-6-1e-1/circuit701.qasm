OPENQASM 2.0;
include "qelib1.inc";
qreg q702[3];
rx(pi) q702[2];
cx q702[2],q702[1];
cx q702[1],q702[0];
rx(pi/4) q702[1];
