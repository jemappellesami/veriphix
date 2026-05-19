OPENQASM 2.0;
include "qelib1.inc";
qreg q760[3];
cx q760[1],q760[0];
cx q760[1],q760[2];
rx(5*pi/4) q760[0];
cx q760[1],q760[0];
