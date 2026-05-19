OPENQASM 2.0;
include "qelib1.inc";
qreg q956[3];
cx q956[1],q956[0];
rx(pi/4) q956[1];
rx(3*pi/2) q956[0];
cx q956[2],q956[1];
cx q956[0],q956[1];
