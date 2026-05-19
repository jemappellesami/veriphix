OPENQASM 2.0;
include "qelib1.inc";
qreg q618[3];
rx(pi/2) q618[2];
cx q618[2],q618[1];
cx q618[0],q618[1];
rx(pi/4) q618[1];
