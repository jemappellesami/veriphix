OPENQASM 2.0;
include "qelib1.inc";
qreg q761[3];
rx(5*pi/4) q761[0];
cx q761[0],q761[1];
cx q761[2],q761[1];
cx q761[1],q761[0];
rx(pi/4) q761[1];
