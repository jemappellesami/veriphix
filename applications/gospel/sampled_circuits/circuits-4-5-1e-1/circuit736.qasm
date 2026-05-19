OPENQASM 2.0;
include "qelib1.inc";
qreg q737[4];
rx(3*pi/2) q737[3];
cx q737[2],q737[3];
cx q737[2],q737[1];
cx q737[1],q737[0];
