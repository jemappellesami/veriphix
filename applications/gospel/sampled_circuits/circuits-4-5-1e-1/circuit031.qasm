OPENQASM 2.0;
include "qelib1.inc";
qreg q32[4];
rx(pi/2) q32[3];
cx q32[2],q32[3];
cx q32[2],q32[1];
cx q32[0],q32[1];
