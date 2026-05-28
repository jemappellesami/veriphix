OPENQASM 2.0;
include "qelib1.inc";
qreg q936[3];
rx(7*pi/4) q936[2];
cx q936[1],q936[2];
cx q936[1],q936[0];
