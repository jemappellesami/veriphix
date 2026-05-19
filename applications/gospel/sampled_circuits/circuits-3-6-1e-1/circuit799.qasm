OPENQASM 2.0;
include "qelib1.inc";
qreg q800[3];
rx(pi) q800[2];
cx q800[1],q800[2];
cx q800[1],q800[0];
rx(pi/4) q800[1];
