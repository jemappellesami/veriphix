OPENQASM 2.0;
include "qelib1.inc";
qreg q429[6];
cx q429[4],q429[3];
cx q429[3],q429[2];
cx q429[1],q429[2];
cx q429[0],q429[1];
rx(pi/4) q429[1];
