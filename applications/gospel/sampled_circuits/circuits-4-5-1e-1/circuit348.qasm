OPENQASM 2.0;
include "qelib1.inc";
qreg q349[4];
cx q349[0],q349[1];
rx(3*pi/2) q349[1];
cx q349[1],q349[2];
cx q349[0],q349[1];
