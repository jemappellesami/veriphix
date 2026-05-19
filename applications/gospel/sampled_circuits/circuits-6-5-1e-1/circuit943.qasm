OPENQASM 2.0;
include "qelib1.inc";
qreg q944[6];
cx q944[1],q944[0];
cx q944[0],q944[1];
cx q944[1],q944[2];
rx(pi/4) q944[0];
