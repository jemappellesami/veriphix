OPENQASM 2.0;
include "qelib1.inc";
qreg q748[4];
cx q748[0],q748[1];
cx q748[3],q748[2];
rx(pi/2) q748[1];
cx q748[2],q748[1];
cx q748[0],q748[1];
