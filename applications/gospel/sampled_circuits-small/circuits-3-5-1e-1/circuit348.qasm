OPENQASM 2.0;
include "qelib1.inc";
qreg q349[3];
cx q349[0],q349[1];
rx(5*pi/4) q349[2];
cx q349[2],q349[1];
cx q349[1],q349[0];
