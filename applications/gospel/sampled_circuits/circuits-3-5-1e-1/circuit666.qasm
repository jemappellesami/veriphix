OPENQASM 2.0;
include "qelib1.inc";
qreg q667[3];
rx(pi/4) q667[2];
cx q667[1],q667[2];
cx q667[0],q667[1];
