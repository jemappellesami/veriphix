OPENQASM 2.0;
include "qelib1.inc";
qreg q429[3];
cx q429[1],q429[0];
rz(7*pi/4) q429[1];
rx(7*pi/4) q429[0];
cx q429[1],q429[0];
cx q429[1],q429[2];
