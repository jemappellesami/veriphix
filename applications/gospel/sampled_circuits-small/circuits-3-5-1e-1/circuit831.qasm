OPENQASM 2.0;
include "qelib1.inc";
qreg q832[3];
rx(3*pi/2) q832[1];
cx q832[1],q832[0];
cx q832[1],q832[2];
cx q832[0],q832[1];
