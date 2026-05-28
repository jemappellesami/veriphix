OPENQASM 2.0;
include "qelib1.inc";
qreg q871[3];
rx(pi/2) q871[2];
cx q871[2],q871[1];
cx q871[0],q871[1];
rx(pi/4) q871[1];
