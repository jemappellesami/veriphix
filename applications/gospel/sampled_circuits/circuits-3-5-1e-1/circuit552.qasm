OPENQASM 2.0;
include "qelib1.inc";
qreg q553[3];
rx(3*pi/4) q553[2];
cx q553[2],q553[1];
cx q553[1],q553[0];
